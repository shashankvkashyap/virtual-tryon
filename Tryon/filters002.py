import subprocess
import keyboard  # pip install keyboard

def run_script(script):
    # Start a subprocess to run the given script
    return subprocess.Popen(['python', script])

def main():
    current_script = None
    scripts = {
        'g': 'glass_virtual_tryon.py',
        'h': 'testhat002.py',
        'm': 'testmask001.py'
    }

    print("Press 'g' for glasses filter, 'h' for hat filter, 'm' for mask filter. Press 'q' to quit.")

    while True:
        try:
            if keyboard.is_pressed('q'):
                break
            
            for key, script in scripts.items():
                if keyboard.is_pressed(key):
                    # Check if a different script is already running
                    if current_script and current_script.poll() is None:
                        current_script.terminate()
                    
                    # Start the selected script
                    current_script = run_script(script)
                    
                    # Wait for key release to avoid repeated triggering
                    while keyboard.is_pressed(key):
                        pass

        except KeyboardInterrupt:
            break

    # Terminate the current script before exiting
    if current_script and current_script.poll() is None:
        current_script.terminate()

if __name__ == "__main__":
    main()
