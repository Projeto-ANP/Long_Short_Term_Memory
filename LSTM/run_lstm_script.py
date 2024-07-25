import sys
from long_short_term_memory import run_lstm_in_thread

def main(horizon, window, epochs, verbose, bool_save, save_model, type_predictions):
    run_lstm_in_thread(horizon=horizon, window=window, epochs=epochs, verbose=verbose, bool_save=bool_save, save_model=False, type_predictions=type_predictions)

if __name__ == "__main__":
    if len(sys.argv) != 8:
        print("Usage: python run_lstm_script.py <horizon> <window> <epochs> <verbose> <bool_save> <save_model> <type_predictions>")
        sys.exit(1)

    horizon = int(sys.argv[1])
    window = int(sys.argv[2])
    epochs = int(sys.argv[3])
    verbose = int(sys.argv[4])
    bool_save = sys.argv[5]
    save_model = sys.argv[6]
    type_predictions = str(sys.argv[7].lower())

    main(horizon, window, epochs, verbose, bool_save, save_model, type_predictions)