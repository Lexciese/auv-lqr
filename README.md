# auv-lqr

## LQR Control for Autonomous Underwater Vehicle (AUV) in Python

The project contains a Linear Quadratic Regulator (LQR) for Autonomous Underwater Vehicle (AUV) implemented in Python. This project is inspired from [juansrojas/lqr-control-auv](https://github.com/juansrojas/lqr-control-auv/) MATLAB codes.

## Run the Program

1. Clone the repository

    ```bash
    git clone https://github.com/Lexciese/auv-lqr.git
    ```

    ```bash
    cd auv-lqr
    ```

2. Install the dependencies

    ```bash
    pip install -r requirements.txt
    ```

3. Run

    ```bash
    python3 main.py
    ```

## Tune the Parameters

- You can tune the parameters in `config.py` which contains every variable for the vehicle dynamics

- Tune the LQR constant, Q element and R element, in `main.py`

- Setpoint or target can be changed in `main.py` at `target_state` variable