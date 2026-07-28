#include <cstddef>
#include <cstdint>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

#include <pybind11/numpy.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include "L2Synchronizer.hpp"
#include "MarketEngine.hpp"

namespace py = pybind11;

namespace {

void validate_replay_metadata(const std::string& file_path) {
    py::module_ qbin = py::module_::import("qbin");
    py::object metadata = qbin.attr("read_metadata")(file_path);
    py::object completeness = metadata.attr("data_complete");
    if (!completeness.is_none() && !completeness.cast<bool>()) {
        throw std::runtime_error(
            "Refusing to replay a recording marked incomplete. "
            "Use the Python diagnostic tools for explicit incomplete-data inspection.");
    }
}

std::vector<quant::l2::Level> parse_python_levels(const py::iterable& values) {
    std::vector<quant::l2::Level> levels;
    for (const py::handle item : values) {
        const py::sequence pair = py::reinterpret_borrow<py::sequence>(item);
        if (py::len(pair) != 2) {
            throw std::invalid_argument("Each L2 level must contain price and quantity.");
        }
        const auto price = pair[0].cast<std::int64_t>();
        const auto quantity = pair[1].cast<std::uint64_t>();
        const quant::l2::Level level{price, quantity};
        quant::l2::validate_level(level);
        levels.push_back(level);
    }
    return levels;
}

py::list levels_to_python(const std::vector<quant::l2::Level>& levels) {
    py::list output;
    for (const auto& level : levels) {
        output.append(py::make_tuple(level.price, level.quantity));
    }
    return output;
}

class NativeL2Synchronizer {
public:
    explicit NativeL2Synchronizer(std::size_t maximum_buffered_events)
        : synchronizer_(maximum_buffered_events) {}

    std::string ingest(
        std::uint64_t receipt_timestamp_ns,
        std::uint64_t event_time_ms,
        std::uint64_t first_update_id,
        std::uint64_t final_update_id,
        const py::iterable& bids,
        const py::iterable& asks) {
        quant::l2::DepthUpdate update{
            receipt_timestamp_ns,
            event_time_ms,
            first_update_id,
            final_update_id,
            parse_python_levels(bids),
            parse_python_levels(asks),
        };
        py::gil_scoped_release release;
        return std::string{quant::l2::to_string(synchronizer_.ingest(update))};
    }

    py::dict install_snapshot(
        std::uint64_t receipt_timestamp_ns,
        std::uint64_t last_update_id,
        const py::iterable& bids,
        const py::iterable& asks) {
        quant::l2::Snapshot snapshot{
            receipt_timestamp_ns,
            last_update_id,
            parse_python_levels(bids),
            parse_python_levels(asks),
        };
        quant::l2::SnapshotInstallResult result;
        {
            py::gil_scoped_release release;
            result = synchronizer_.install_snapshot(snapshot);
        }
        py::dict output;
        output["result"] = std::string{quant::l2::to_string(result.result)};
        output["stale_events"] = result.stale_events;
        output["applied_events"] = result.applied_events;
        output["first_applied_buffer_index"] = result.first_applied_buffer_index;
        return output;
    }

    void reset() noexcept { synchronizer_.reset(); }

    [[nodiscard]] std::string state() const {
        return std::string{quant::l2::to_string(synchronizer_.state())};
    }

    [[nodiscard]] std::size_t buffered_events() const noexcept {
        return synchronizer_.buffered_events();
    }

    [[nodiscard]] std::uint64_t last_update_id() const noexcept {
        return synchronizer_.book().last_update_id();
    }

    [[nodiscard]] std::uint64_t state_hash() const noexcept {
        return synchronizer_.book().state_hash();
    }

    [[nodiscard]] py::object best_bid() const {
        if (synchronizer_.book().all_bids().empty()) {
            return py::none();
        }
        const auto& level = synchronizer_.book().best_bid();
        return py::make_tuple(level.price, level.quantity);
    }

    [[nodiscard]] py::object best_ask() const {
        if (synchronizer_.book().all_asks().empty()) {
            return py::none();
        }
        const auto& level = synchronizer_.book().best_ask();
        return py::make_tuple(level.price, level.quantity);
    }

    [[nodiscard]] py::dict top_levels(std::size_t limit) const {
        py::dict output;
        output["bids"] = levels_to_python(synchronizer_.book().bids(limit));
        output["asks"] = levels_to_python(synchronizer_.book().asks(limit));
        return output;
    }

    [[nodiscard]] std::uint64_t quantity_at(
        const std::string& side,
        std::int64_t price) const {
        if (side == "bid") {
            return synchronizer_.book().quantity_at(true, price);
        }
        if (side == "ask") {
            return synchronizer_.book().quantity_at(false, price);
        }
        throw std::invalid_argument("Side must be 'bid' or 'ask'.");
    }

private:
    quant::l2::Synchronizer synchronizer_;
};

}  // namespace

PYBIND11_MODULE(quant_engine, module) {
    module.doc() =
        "Live/replay top-of-book engine plus native sequence-correct L2 book";

    PYBIND11_NUMPY_DTYPE(
        OrderBookState,
        timestamp_ns,
        best_bid,
        best_ask,
        bid_volume,
        ask_volume);

    module.attr("order_book_dtype") = py::dtype::of<OrderBookState>();
    module.attr("binary_version") = quant::BinaryVersion;
    module.attr("binary_volume_scale") = quant::BinaryVolumeScale;
    module.attr("l2_price_scale") = quant::l2::PriceScale;
    module.attr("l2_quantity_scale") = quant::l2::QuantityScale;

    module.def(
        "l2_parse_price",
        [](const std::string& value) {
            std::int64_t output = 0;
            if (!quant::l2::parse_price(value, output)) {
                throw std::invalid_argument("Invalid L2 fixed-point price.");
            }
            return output;
        });
    module.def(
        "l2_parse_quantity",
        [](const std::string& value) {
            std::uint64_t output = 0;
            if (!quant::l2::parse_quantity(value, output)) {
                throw std::invalid_argument("Invalid L2 fixed-point quantity.");
            }
            return output;
        });

    py::class_<NativeL2Synchronizer>(module, "L2Synchronizer")
        .def(py::init<std::size_t>(), py::arg("maximum_buffered_events") = 200'000)
        .def(
            "ingest",
            &NativeL2Synchronizer::ingest,
            py::arg("receipt_timestamp_ns"),
            py::arg("event_time_ms"),
            py::arg("first_update_id"),
            py::arg("final_update_id"),
            py::arg("bids"),
            py::arg("asks"))
        .def(
            "install_snapshot",
            &NativeL2Synchronizer::install_snapshot,
            py::arg("receipt_timestamp_ns"),
            py::arg("last_update_id"),
            py::arg("bids"),
            py::arg("asks"))
        .def("reset", &NativeL2Synchronizer::reset)
        .def_property_readonly("state", &NativeL2Synchronizer::state)
        .def_property_readonly("buffered_events", &NativeL2Synchronizer::buffered_events)
        .def_property_readonly("last_update_id", &NativeL2Synchronizer::last_update_id)
        .def_property_readonly("state_hash", &NativeL2Synchronizer::state_hash)
        .def_property_readonly("best_bid", &NativeL2Synchronizer::best_bid)
        .def_property_readonly("best_ask", &NativeL2Synchronizer::best_ask)
        .def("top_levels", &NativeL2Synchronizer::top_levels, py::arg("limit") = 20)
        .def("quantity_at", &NativeL2Synchronizer::quantity_at);

    py::class_<IngestionEngine>(module, "IngestionEngine")
        .def(py::init<>())
        .def("start", &IngestionEngine::start)
        .def(
            "start_live",
            &IngestionEngine::start_live,
            py::arg("recording_path") = "")
        .def(
            "start_replay",
            [](IngestionEngine& engine,
               const std::string& file_path,
               double speed,
               bool allow_incomplete) {
                if (!allow_incomplete) {
                    validate_replay_metadata(file_path);
                }
                engine.start_replay(file_path, speed);
            },
            py::arg("file_path"),
            py::arg("speed") = 1.0,
            py::arg("allow_incomplete") = false)
        .def(
            "stop",
            [](IngestionEngine& engine) {
                py::gil_scoped_release release;
                engine.stop();
            })
        .def("is_running", &IngestionEngine::is_running)
        .def("state", &IngestionEngine::state_name)
        .def("last_error", &IngestionEngine::last_error)
        .def("now_ns", &IngestionEngine::now_ns)
        .def("dropped_ticks", &IngestionEngine::dropped_ticks)
        .def("malformed_messages", &IngestionEngine::malformed_messages)
        .def("reconnect_count", &IngestionEngine::reconnect_count)
        .def("recorded_ticks", &IngestionEngine::recorded_ticks)
        .def("recording_accepted", &IngestionEngine::recording_accepted)
        .def("recording_dropped", &IngestionEngine::recording_dropped)
        .def("recording_write_errors", &IngestionEngine::recording_write_errors)
        .def("replayed_ticks", &IngestionEngine::replayed_ticks)
        .def("replay_backpressure_events", &IngestionEngine::replay_backpressure_events)
        .def("replay_errors", &IngestionEngine::replay_errors)
        .def(
            "consume_batch",
            [](IngestionEngine& engine,
               py::array_t<OrderBookState, py::array::c_style> array)
                -> std::size_t {
                if (array.ndim() != 1) {
                    throw std::runtime_error(
                        "Destination array must be one-dimensional.");
                }
                if (!array.writeable()) {
                    throw std::runtime_error(
                        "Destination array must be writable.");
                }
                OrderBookState* destination = array.mutable_data();
                const std::size_t maximum_count =
                    static_cast<std::size_t>(array.shape(0));
                py::gil_scoped_release release;
                return engine.consume_batch(destination, maximum_count);
            },
            py::arg("array"));
}
