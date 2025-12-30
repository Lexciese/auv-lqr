# auv-lqr

## LQR Control for Autonomous Underwater Vehicle (AUV) in Python

This repository present a Linear Quadratic Regulator (LQR) for Autonomous Underwater Vehicle (AUV) implemented in Python. This works is inspired from [juansrojas/lqr-control-auv](https://github.com/juansrojas/lqr-control-auv/) MATLAB codes.  

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

## References

- juansrojas. (2022).  
  *github.com/juansrojas/lqr-control-auv: LQR Control of an Autonomous Underwater Vehicle*  
  Available online at [juansrojas/lqr-control-auv](https://github.com/juansrojas/lqr-control-auv/), accessed 2025

- Fossen, T. I. (2021).  
  *Fossen’s Marine Craft Model.*  
  Available online at [https://fossen.biz/html/marineCraftModel.html](https://fossen.biz/html/marineCraftModel.html), accessed 2025.  

- Chin, C. S. (2017).  
  *Computer-Aided Control Systems Design (p. 138). CRC Press.*

- Fossen, T. I. (2011).  
  *Handbook of Marine Craft Hydrodynamics and Motion Control (p. 60). John Wiley & Sons.*

