
#pragma once

#include <algorithm>
#include <atomic>
#include <charconv>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <iostream>
#include <limits>
#include <mutex>
#include <optional>
#include <string>
#include <string_view>
#include <system_error>
#include <thread>

#include <boost/asio/connect.hpp>
#include <boost/asio/ip/tcp.hpp>
#include <boost/asio/ssl/context.hpp>
#include <boost/asio/ssl/host_name_verification.hpp>
#include <boost/asio/steady_timer.hpp>
#include <boost/beast/core.hpp>
#include <boost/beast/http.hpp>
#include <boost/beast/ssl.hpp>
#include <boost/beast/version.hpp>
#include <boost/beast/websocket.hpp>
#include <boost/beast/websocket/ssl.hpp>
#include <openssl/err.h>
#include <openssl/ssl.h>
#include <simdjson.h>

#include "BinaryRecorder.hpp"
#include "RingBuffer.hpp"

namespace beast = boost::beast;
namespace http = beast::http;
namespace websocket = beast::websocket;
namespace net = boost::asio;
namespace ssl = boost::asio::ssl;
using tcp = net::ip::tcp;

class BinanceFeed {
public:
    BinanceFeed(
        SPSCRingBuffer<65536>& ring_buffer,
        std::atomic<std::uint64_t>& dropped_ticks,
        std::atomic<std::uint64_t>& malformed_messages,
        std::atomic<std::uint64_t>& reconnect_count,
        BinaryRecorder* recorder)
        : ring_buffer_(ring_buffer),
          dropped_ticks_(dropped_ticks),
          malformed_messages_(malformed_messages),
          reconnect_count_(reconnect_count),
          recorder_(recorder) {}

    void request_stop() noexcept {
        std::lock_guard lock(operation_mutex_);
        if (active_resolver_ != nullptr) {
            active_resolver_->cancel();
        }
        if (active_stream_ != nullptr) {
            beast::error_code ignored;
            active_stream_->socket().cancel(ignored);
            active_stream_->socket().shutdown(tcp::socket::shutdown_both, ignored);
            active_stream_->socket().close(ignored);
        }
    }

    std::string last_error() const {
        std::lock_guard lock(error_mutex_);
        return last_error_;
    }

    void run(std::atomic<bool>& running) noexcept {
        std::chrono::milliseconds reconnect_delay{500};

        std::optional<std::uint64_t> previous_update_id;
        while (running.load(std::memory_order_acquire)) {
            try {
                run_session(running, previous_update_id);
                reconnect_delay = std::chrono::milliseconds{500};
            } catch (const std::exception& exception) {
                clear_active_operations();
                if (!running.load(std::memory_order_acquire)) {
                    break;
                }

                set_last_error(exception.what());
                reconnect_count_.fetch_add(1, std::memory_order_relaxed);
                std::cerr << "[Binance Feed] Connection error: "
                          << exception.what() << '\n';
                wait_before_reconnect(running, reconnect_delay);
                reconnect_delay = std::min(
                    reconnect_delay * 2,
                    std::chrono::milliseconds{10'000});
            } catch (...) {
                clear_active_operations();
                if (!running.load(std::memory_order_acquire)) {
                    break;
                }
                set_last_error("Unknown connection error.");
                reconnect_count_.fetch_add(1, std::memory_order_relaxed);
                std::cerr << "[Binance Feed] Unknown connection error.\n";
                wait_before_reconnect(running, reconnect_delay);
                reconnect_delay = std::min(
                    reconnect_delay * 2,
                    std::chrono::milliseconds{10'000});
            }
        }
        clear_active_operations();
    }

private:
    using WebSocketStream = websocket::stream<beast::ssl_stream<beast::tcp_stream>>;
    static constexpr double VolumeScale = 1'000'000.0;

    SPSCRingBuffer<65536>& ring_buffer_;
    std::atomic<std::uint64_t>& dropped_ticks_;
    std::atomic<std::uint64_t>& malformed_messages_;
    std::atomic<std::uint64_t>& reconnect_count_;
    BinaryRecorder* recorder_;

    std::mutex operation_mutex_;
    mutable std::mutex error_mutex_;
    std::string last_error_;
    tcp::resolver* active_resolver_{nullptr};
    beast::tcp_stream* active_stream_{nullptr};

    class ActiveOperationGuard {
    public:
        explicit ActiveOperationGuard(BinanceFeed& owner) noexcept
            : owner_(owner) {}
        ActiveOperationGuard(const ActiveOperationGuard&) = delete;
        ActiveOperationGuard& operator=(const ActiveOperationGuard&) = delete;
        ~ActiveOperationGuard() { owner_.clear_active_operations(); }

    private:
        BinanceFeed& owner_;
    };

    void set_last_error(const std::string& message) noexcept {
        try {
            std::lock_guard lock(error_mutex_);
            last_error_ = message;
        } catch (...) {
        }
    }

    void clear_last_error() noexcept {
        try {
            std::lock_guard lock(error_mutex_);
            last_error_.clear();
        } catch (...) {
        }
    }

    static std::uint64_t monotonic_time_ns() noexcept {
        const auto now = std::chrono::steady_clock::now();
        return static_cast<std::uint64_t>(
            std::chrono::duration_cast<std::chrono::nanoseconds>(
                now.time_since_epoch()).count());
    }

    void clear_active_operations() noexcept {
        std::lock_guard lock(operation_mutex_);
        active_resolver_ = nullptr;
        active_stream_ = nullptr;
    }

    void register_operations(
        tcp::resolver& resolver,
        beast::tcp_stream& stream) noexcept {
        std::lock_guard lock(operation_mutex_);
        active_resolver_ = &resolver;
        active_stream_ = &stream;
    }

    static void wait_before_reconnect(
        const std::atomic<bool>& running,
        std::chrono::milliseconds delay) noexcept {
        constexpr auto PollInterval = std::chrono::milliseconds{100};
        auto remaining = delay;
        while (remaining.count() > 0 &&
               running.load(std::memory_order_acquire)) {
            const auto sleep_duration = std::min(remaining, PollInterval);
            std::this_thread::sleep_for(sleep_duration);
            remaining -= sleep_duration;
        }
    }

    static bool parse_double(std::string_view text, double& value) noexcept {
        if (text.empty()) {
            return false;
        }
        const char* begin = text.data();
        const char* end = begin + text.size();
        const auto result = std::from_chars(
            begin, end, value, std::chars_format::general);
        return result.ec == std::errc{} && result.ptr == end &&
               std::isfinite(value);
    }

    static bool parse_scaled_volume(
        std::string_view text,
        std::uint32_t& output) noexcept {
        double quantity = 0.0;
        if (!parse_double(text, quantity) || quantity < 0.0) {
            return false;
        }
        const double scaled = quantity * VolumeScale;
        const double maximum = static_cast<double>(
            std::numeric_limits<std::uint32_t>::max());
        if (!std::isfinite(scaled) || scaled > maximum) {
            return false;
        }
        output = static_cast<std::uint32_t>(std::llround(scaled));
        return true;
    }

    bool parse_and_publish(
        simdjson::dom::parser& parser,
        const std::string& message,
        std::uint64_t receipt_timestamp_ns,
        std::optional<std::uint64_t>& previous_update_id) noexcept {
        try {
            simdjson::padded_string padded_message{message};
            simdjson::dom::element document;
            if (parser.parse(padded_message).get(document) != simdjson::SUCCESS) {
                return false;
            }

            std::string_view bid_text;
            std::string_view bid_volume_text;
            std::string_view ask_text;
            std::string_view ask_volume_text;
            std::string_view symbol;
            std::uint64_t update_id = 0;

            if (document["b"].get_string().get(bid_text) != simdjson::SUCCESS ||
                document["B"].get_string().get(bid_volume_text) != simdjson::SUCCESS ||
                document["a"].get_string().get(ask_text) != simdjson::SUCCESS ||
                document["A"].get_string().get(ask_volume_text) != simdjson::SUCCESS ||
                document["s"].get_string().get(symbol) != simdjson::SUCCESS ||
                document["u"].get_uint64().get(update_id) != simdjson::SUCCESS) {
                return false;
            }
            if (symbol != "BTCUSDT" ||
                (previous_update_id.has_value() &&
                 update_id <= previous_update_id.value())) {
                return false;
            }

            double best_bid = 0.0;
            double best_ask = 0.0;
            std::uint32_t bid_volume = 0;
            std::uint32_t ask_volume = 0;
            if (!parse_double(bid_text, best_bid) ||
                !parse_double(ask_text, best_ask) ||
                !parse_scaled_volume(bid_volume_text, bid_volume) ||
                !parse_scaled_volume(ask_volume_text, ask_volume)) {
                return false;
            }

            const OrderBookState state{
                receipt_timestamp_ns,
                best_bid,
                best_ask,
                bid_volume,
                ask_volume};
            if (!valid_order_book_state(state)) {
                return false;
            }

            previous_update_id = update_id;
            if (recorder_ != nullptr) {
                recorder_->record(state, update_id);
            }
            if (!ring_buffer_.push(state)) {
                dropped_ticks_.fetch_add(1, std::memory_order_relaxed);
            }
            return true;
        } catch (...) {
            return false;
        }
    }

    static tcp::resolver::results_type resolve_with_timeout(
        net::io_context& io_context,
        tcp::resolver& resolver,
        const std::string& host,
        const std::string& port,
        const std::atomic<bool>& running) {
        beast::error_code resolve_error = net::error::would_block;
        std::optional<tcp::resolver::results_type> endpoints;
        net::steady_timer timer{io_context};
        timer.expires_after(std::chrono::seconds{10});

        resolver.async_resolve(
            host,
            port,
            [&](const beast::error_code& error,
                tcp::resolver::results_type result) {
                resolve_error = error;
                if (!error) {
                    endpoints = std::move(result);
                }
                timer.cancel();
            });

        timer.async_wait([&](const beast::error_code& error) {
            if (!error) {
                resolver.cancel();
            }
        });

        while (running.load(std::memory_order_acquire) &&
               resolve_error == net::error::would_block) {
            io_context.run_for(std::chrono::milliseconds{100});
            io_context.restart();
        }
        if (!running.load(std::memory_order_acquire)) {
            resolver.cancel();
            throw std::runtime_error("Feed stop requested during DNS resolution.");
        }
        if (resolve_error) {
            throw beast::system_error(resolve_error, "DNS resolution failed");
        }
        if (!endpoints.has_value()) {
            throw std::runtime_error("DNS resolution returned no endpoints.");
        }
        return std::move(endpoints.value());
    }

    void run_session(
        std::atomic<bool>& running,
        std::optional<std::uint64_t>& previous_update_id) {
        const std::string host = "stream.binance.com";
        const std::string port = "9443";
        const std::string target = "/ws/btcusdt@bookTicker";

        net::io_context io_context;
        ssl::context ssl_context{ssl::context::tls_client};
        ssl_context.set_options(
            ssl::context::default_workarounds |
            ssl::context::no_sslv2 |
            ssl::context::no_sslv3);
        ssl_context.set_default_verify_paths();
        ssl_context.set_verify_mode(ssl::verify_peer);

        tcp::resolver resolver{io_context};
        WebSocketStream websocket_stream{io_context, ssl_context};
        auto& tcp_stream = beast::get_lowest_layer(websocket_stream);
        register_operations(resolver, tcp_stream);
        ActiveOperationGuard operation_guard{*this};

        if (!SSL_set_tlsext_host_name(
                websocket_stream.next_layer().native_handle(), host.c_str())) {
            const beast::error_code error{
                static_cast<int>(::ERR_get_error()),
                net::error::get_ssl_category()};
            throw beast::system_error(error, "Failed to configure TLS SNI");
        }
        websocket_stream.next_layer().set_verify_callback(
            ssl::host_name_verification{host});

        const auto endpoints = resolve_with_timeout(
            io_context, resolver, host, port, running);
        tcp_stream.expires_after(std::chrono::seconds{15});
        tcp_stream.connect(endpoints);
        websocket_stream.next_layer().handshake(ssl::stream_base::client);
        tcp_stream.expires_never();

        auto timeout = websocket::stream_base::timeout::suggested(
            beast::role_type::client);
        timeout.handshake_timeout = std::chrono::seconds{15};
        timeout.idle_timeout = std::chrono::seconds{5};
        timeout.keep_alive_pings = true;
        websocket_stream.set_option(timeout);
        websocket_stream.set_option(websocket::stream_base::decorator{
            [](websocket::request_type& request) {
                request.set(
                    http::field::user_agent,
                    std::string{"market-systems/1.0 "} +
                        BOOST_BEAST_VERSION_STRING);
            }});
        websocket_stream.handshake(host + ":" + port, target);
        clear_last_error();

        if (recorder_ != nullptr) {
            recorder_->mark_session_boundary("connection_start");
        }
        std::cerr << "[Binance Feed] Connected to " << host << target << '\n';

        beast::flat_buffer network_buffer;
        simdjson::dom::parser parser;

        while (running.load(std::memory_order_acquire)) {
            websocket_stream.read(network_buffer);
            const std::uint64_t receipt_timestamp_ns = monotonic_time_ns();

            if (!websocket_stream.got_text()) {
                network_buffer.consume(network_buffer.size());
                continue;
            }
            const std::string message = beast::buffers_to_string(
                network_buffer.data());
            network_buffer.consume(network_buffer.size());

            if (!parse_and_publish(
                    parser,
                    message,
                    receipt_timestamp_ns,
                    previous_update_id)) {
                malformed_messages_.fetch_add(1, std::memory_order_relaxed);
            }
        }

        beast::error_code close_error;
        websocket_stream.close(websocket::close_code::normal, close_error);
    }
};

