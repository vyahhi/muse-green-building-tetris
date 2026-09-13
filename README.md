# Muse 2 Green Building Tetris

Use a Muse 2 headband to play Tetris on a 17×9 RGB façade simulator. Fused
accelerometer/gyroscope orientation continuously selects a column; a calibrated
down-and-back nod rotates the piece. The repository also includes a fixed-scale
live EEG viewer and experimental personalized wink detection.

## Setup

Install [uv](https://docs.astral.sh/uv/), then sync the environment:

```bash
uv sync
cp .env.example .env
```

Edit `.env` with your Muse's macOS Bluetooth identifier and display settings.
The `.env` file and personal calibration profiles are ignored by Git.

```dotenv
MUSE_ADDRESS=your-muse-bluetooth-identifier
GREEN_BUILDING_API_ROOT=https://example.com/api
GREEN_BUILDING_DISPLAY_ID=your-display-id
```

## Live EEG viewer

Power on the Muse 2 and make sure it is not connected to the Muse phone app.
Muse uses Bluetooth Low Energy and normally should **not** be paired manually in
macOS Bluetooth Settings.

Run:

```bash
UV_CACHE_DIR=.uv-cache uv run live_view.py
```

On first use, macOS may ask whether the calling app can use Bluetooth. Allow it.
The viewer displays the four raw EEG channels: TP9, AF7, AF8, and TP10.
The Y axis stays fixed at ±500 µV. To use another fixed range, for example
±250 µV, run `UV_CACHE_DIR=.uv-cache uv run live_view.py --y-limit 250`.

## Play Tetris

Power on the headband, close other Muse apps, and run:

```bash
UV_CACHE_DIR=.uv-cache uv run tetris_muselsl.py
```

- Lean/tilt left or right: continuously choose the target column
- Quick deliberate nod down and back: rotate once
- Pieces fall automatically

Use `--rotation blink` to try the personalized double-blink detector instead,
or `--rotation off` to disable rotation input. Blink mode first requires a local
`wink_profile.json` created by running `uv run calibrate_winks.py`; that personal
calibration file is intentionally excluded from Git.

At startup, follow the façade calibration cues: remain centered, sweep your
head tilt left/right twice, then perform three slow down-and-back nods.
