import subprocess
import multiprocessing
from long_short_term_memory import product_and_single_thread_testing

if __name__ == "__main__":
    try:

        multiprocessing.set_start_method("spawn")
        
        # # Test Univariate - PR GLP
        thread = multiprocessing.Process(target=product_and_single_thread_testing)
        thread.start()
        thread.join()

    
        # # Define the parameters for each loop_lstm call
        # lstm_params = [
        #     (12, 3, 200, 1, True, False, 'direct'),
        #     (12, 6, 200, 1, True, False, 'direct'),
        #     (12, 12, 200, 1, True, False, 'direct'),
        #     (12, 24, 200, 1, True, False, 'direct'),
        #     (12, 36, 200, 1, True, False, 'direct')
        # ]
        
        # processes = []
        
        # # Start a new process for each set of parameters
        # for params in lstm_params:
        #     cmd = [
        #         "python", "run_lstm_script.py",
        #         str(params[0]), str(params[1]), str(params[2]),
        #         str(params[3]), str(params[4]), str(params[5]), str(params[6])
        #     ]
        #     p = subprocess.Popen(cmd)
        #     processes.append(p)
        
        # # Wait for all processes to complete
        # for p in processes:
        #     p.wait()

    except Exception as e:
        print("An error occurred:", e)


# Direct dense 12
# RMSE: 0.24700138746383068
# MAPE: 1.0121369793373287
# PBE: 93.6650372080749
# POCID: 63.63636363636363

# Rescaled results:
# Rescaled RMSE: 2518.379215155715
# Rescaled MAPE: 2.012186331353577
# Rescaled PBE: 1.3275965390074267
# Rescaled POCID: 63.63636363636363
# Rescaled MASE: 0.6803182175946246

#  Direct dense 1 
# RMSE: 0.16318518038520935
# MAPE: 0.9237601048259081
# PBE: 1109.333848416835
# POCID: 90.9090909090909

# Rescaled results:
# Rescaled RMSE: 1638.2928055334114
# Rescaled MAPE: 1.4614575433092136
# Rescaled PBE: 1.3087544992595794
# Rescaled POCID: 90.9090909090909
# Rescaled MASE: 0.48237240356953

# Recursive
# RMSE: 0.257434768611349
# MAPE: 1.645812601794045
# PBE: 1364.8869288889998
# POCID: 63.63636363636363

# Rescaled results:
# Rescaled RMSE: 2645.727683394386
# Rescaled MAPE: 2.567752866289212
# Rescaled PBE: 1.6375094274428472
# Rescaled POCID: 63.63636363636363
# Rescaled MASE: 0.8491199584066879