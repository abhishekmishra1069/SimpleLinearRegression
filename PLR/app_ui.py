"""
app_ui.py
A tkinter-based GUI for polynomial regression salary prediction.
Loads the pre-trained model and transformer from pickle files.
Provides real-time prediction updates as the user types experience.
"""

import tkinter as tk
from tkinter import messagebox
import numpy as np
import pickle
import os
import sys

# ============================================================================
# Load the model and transformer at startup (same as app.py)
# ============================================================================
try:
    with open('poly_reg_model.pkl', 'rb') as f:
        model = pickle.load(f)
        print("[✓] poly_reg_model.pkl loaded successfully")
except FileNotFoundError:
    print("[✗] Error: poly_reg_model.pkl not found.")
    print("    Ensure you are in the PLR/ directory and Train_and_save.ipynb was run.")
    sys.exit(1)
except Exception as e:
    print(f"[✗] Error loading poly_reg_model.pkl: {e}")
    sys.exit(1)

try:
    with open('poly_features.pkl', 'rb') as f:
        poly = pickle.load(f)
        print("[✓] poly_features.pkl loaded successfully")
except FileNotFoundError:
    print("[✗] Error: poly_features.pkl not found.")
    print("    Ensure you are in the PLR/ directory and Train_and_save.ipynb was run.")
    sys.exit(1)
except Exception as e:
    print(f"[✗] Error loading poly_features.pkl: {e}")
    sys.exit(1)


# ============================================================================
# GUI Application Class
# ============================================================================
class SalaryPredictorUI:
    """
    A tkinter GUI application that predicts salary based on years of experience.
    - Real-time prediction updates as the user types
    - Input validation to handle non-numeric or empty input gracefully
    - Clear display of results with formatted output
    """

    def __init__(self, root):
        """Initialize the GUI window and widgets."""
        self.root = root
        self.root.title("Polynomial Regression Salary Predictor")
        self.root.geometry("500x300")
        self.root.resizable(False, False)

        # Set a modern-looking background color
        self.root.config(bg="#f0f0f0")

        # ====================================================================
        # Title Label
        # ====================================================================
        title_label = tk.Label(
            self.root,
            text="Salary Prediction Tool",
            font=("Arial", 18, "bold"),
            bg="#f0f0f0",
            fg="#333333"
        )
        title_label.pack(pady=20)

        # ====================================================================
        # Input Frame
        # ====================================================================
        input_frame = tk.Frame(self.root, bg="#f0f0f0")
        input_frame.pack(pady=10)

        # Label for the input field
        experience_label = tk.Label(
            input_frame,
            text="Years of Experience:",
            font=("Arial", 12),
            bg="#f0f0f0",
            fg="#333333"
        )
        experience_label.pack(side=tk.LEFT, padx=5)

        # Entry widget for user input
        # - User types here and we listen for changes with `textvariable`
        self.experience_var = tk.StringVar()
        # Bind the callback to update predictions whenever the text changes
        self.experience_var.trace("w", self.on_experience_change)

        experience_entry = tk.Entry(
            input_frame,
            textvariable=self.experience_var,
            font=("Arial", 12),
            width=15,
            bd=2,
            relief=tk.SUNKEN
        )
        experience_entry.pack(side=tk.LEFT, padx=5)

        # ====================================================================
        # Output Frame
        # ====================================================================
        output_frame = tk.Frame(self.root, bg="#ffffff", bd=1, relief=tk.SOLID)
        output_frame.pack(pady=20, padx=20, fill=tk.BOTH, expand=True)

        # Label for predicted salary
        result_title = tk.Label(
            output_frame,
            text="Predicted Salary:",
            font=("Arial", 12, "bold"),
            bg="#ffffff",
            fg="#333333"
        )
        result_title.pack(anchor=tk.W, padx=10, pady=5)

        # Display area for the predicted salary
        # This label updates in real-time as the user types
        self.salary_label = tk.Label(
            output_frame,
            text="$0.00",
            font=("Arial", 24, "bold"),
            bg="#ffffff",
            fg="#27ae60"  # Green color for positive results
        )
        self.salary_label.pack(anchor=tk.W, padx=10, pady=10)

        # Status/info label for displaying messages
        self.status_label = tk.Label(
            output_frame,
            text="Enter experience to see prediction",
            font=("Arial", 10),
            bg="#ffffff",
            fg="#7f8c8d"  # Gray for informational text
        )
        self.status_label.pack(anchor=tk.W, padx=10, pady=5)

        # ====================================================================
        # Button Frame
        # ====================================================================
        button_frame = tk.Frame(self.root, bg="#f0f0f0")
        button_frame.pack(pady=10)

        # Reset button to clear input and output
        reset_button = tk.Button(
            button_frame,
            text="Reset",
            font=("Arial", 11),
            bg="#e74c3c",
            fg="white",
            padx=15,
            pady=5,
            command=self.reset_form
        )
        reset_button.pack(side=tk.LEFT, padx=5)

        # Exit button to close the application
        exit_button = tk.Button(
            button_frame,
            text="Exit",
            font=("Arial", 11),
            bg="#34495e",
            fg="white",
            padx=15,
            pady=5,
            command=self.root.quit
        )
        exit_button.pack(side=tk.LEFT, padx=5)

    def on_experience_change(self, *args):
        """
        Callback triggered whenever the experience input field changes.
        - Called automatically by tkinter when textvariable is modified
        - Validates input and updates prediction in real-time
        - Handles empty and non-numeric inputs gracefully
        """
        experience_text = self.experience_var.get().strip()

        # If input is empty, reset prediction display to initial state
        if not experience_text:
            self.salary_label.config(text="$0.00", fg="#27ae60")
            self.status_label.config(text="Enter experience to see prediction", fg="#7f8c8d")
            return

        try:
            # Convert input string to float
            # This will raise ValueError if input is not a valid number
            experience = float(experience_text)

            # Validate that experience is non-negative
            if experience < 0:
                self.salary_label.config(text="Invalid", fg="#e74c3c")
                self.status_label.config(text="Experience cannot be negative", fg="#e74c3c")
                return

            # Create a 2D array with concrete dtype (float) for efficient prediction
            # Shape: (1, 1) for a single sample
            input_arr = np.array([[experience]], dtype=float)

            # Transform the input using the fitted PolynomialFeatures transformer
            # This applies the same polynomial expansion used during training
            experience_poly = poly.transform(input_arr)

            # Make the prediction using the pre-trained LinearRegression model
            prediction = model.predict(experience_poly)[0]

            # Ensure prediction is a native Python float (not numpy type)
            prediction = float(prediction)

            # Format the predicted salary as currency and display it
            self.salary_label.config(text=f"${prediction:,.2f}", fg="#27ae60")
            self.status_label.config(text=f"Experience: {experience} years", fg="#7f8c8d")

        except ValueError:
            # Input is not a valid number
            self.salary_label.config(text="Invalid", fg="#e74c3c")
            self.status_label.config(text="Please enter a valid number", fg="#e74c3c")
        except Exception as e:
            # Catch unexpected errors (e.g., model prediction failure)
            self.salary_label.config(text="Error", fg="#e74c3c")
            self.status_label.config(text=f"Prediction error: {str(e)}", fg="#e74c3c")

    def reset_form(self):
        """
        Reset the input field and output display to initial state.
        - Clears the experience input
        - Resets prediction display to $0.00
        - Resets status message
        """
        self.experience_var.set("")
        self.salary_label.config(text="$0.00", fg="#27ae60")
        self.status_label.config(text="Enter experience to see prediction", fg="#7f8c8d")


# ============================================================================
# Main Entry Point
# ============================================================================
if __name__ == '__main__':
    """
    Create the tkinter root window and launch the GUI application.
    - Runs in the local environment with the same model and transformer as app.py
    - Provides an alternative to the REST API for local desktop users
    """
    try:
        print("[*] Initializing GUI...")
        root = tk.Tk()
        app = SalaryPredictorUI(root)
        print("[✓] GUI initialized successfully")
        print("[*] Launching window... (click in the window to activate it)")
        root.mainloop()
        print("[✓] Application closed normally")
    except Exception as e:
        print(f"[✗] Error starting GUI: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
