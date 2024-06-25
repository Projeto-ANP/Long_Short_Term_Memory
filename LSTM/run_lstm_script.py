import sys
from long_short_term_memory import run_lstm_in_thread

def main(horizon, window, epochs, verbose, bool_save, save_model):
    run_lstm_in_thread(horizon=horizon, window=window, epochs=epochs, verbose=verbose, bool_save=bool_save, save_model=save_model)

if __name__ == "__main__":
    if len(sys.argv) != 7:
        print("Usage: python loop_lstm_script.py <horizon> <window> <epochs> <verbose> <bool_save> <save_model>")
        sys.exit(1)

    horizon = int(sys.argv[1])
    window = int(sys.argv[2])
    epochs = int(sys.argv[3])
    verbose = int(sys.argv[4])
    bool_save = sys.argv[5].lower() == 'true'
    save_model = sys.argv[6].lower() == 'true'

    main(horizon, window, epochs, verbose, bool_save, save_model)