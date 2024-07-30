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