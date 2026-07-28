# Repository setup

Recommended GitHub settings:

- **Repository name:** `market-systems-workstation`
- **Description:** `C++20/Python platform for sequence-correct L2 market data, deterministic replay, execution simulation, and microstructure analysis.`
- **Visibility:** Public
- **License:** MIT
- **Topics:** `cpp`, `python`, `market-data`, `order-book`, `websocket`, `concurrency`, `deterministic-replay`, `pybind11`, `microstructure`, `cmake`, `testing`

## First push

Create a new empty GitHub repository without adding a README, license, or `.gitignore`, then run from this folder:

```powershell
git init
git add .
git commit -m "Initial public release"
git branch -M main
git remote add origin <repository-url>
git push -u origin main
```

Do not manufacture earlier commits. Keep later commits focused and descriptive.

## Do not push

The `.gitignore` excludes local environments, native builds, recordings, generated reports, logs, compiled modules, and release archives. Do not force-add those files.

A portable Windows build may be uploaded later as a **GitHub Release asset**. It should not be committed into the source repository.
