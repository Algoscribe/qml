import tkinter as tk
from tkinter import ttk, messagebox
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from simulator import simulate
from utils import ATTACKS
import datetime

# =========================================================
# THEME & DESIGN CONSTANTS (Cyber-Security Palette)
# =========================================================
CLR_BG = "#05070a"        # Deep Space Black
CLR_PANEL = "#0d1117"     # Dark Navy Panel
CLR_ACCENT = "#00f2ff"    # Cyber Cyan
CLR_TEXT = "#c9d1d9"      # Silver Text
CLR_WARNING = "#ff3e3e"   # Crisis Red
CLR_SUCCESS = "#00ff9d"   # Quantum Green

# =========================================================
# METRIC SCIENTIFIC MEANINGS (Layman Explainer)
# =========================================================
metric_guide = """
[HOW TO READ THE QUANTUM METRICS]

1. QBER (Signal Disturbance): 
   Think of this as 'static' on a radio. If an attacker 
   touches the signal, the static increases instantly.

2. KEYLEN (Data Volume): 
   The total amount of secure 'secret code' we made. 
   If this drops, someone is stealing your light particles.

3. BIAS (Data Balance): 
   A fair coin should be 50/50. If the data favors 0s 
   or 1s, the 'digital dice' have been tampered with.

4. ENTROPY (Randomness Quality): 
   This is the 'strength' of your lock. High entropy 
   means the key is impossible to guess. Low means weak.

5. LOSS (Signal Fade): 
   How much light is lost in the wire. High loss 
   usually means someone has physically tapped the line.
"""

# =========================================================
# ATTACK DATABASE (Layman "Story Mode" Descriptions)
# =========================================================
attack_db = {
    "normal": (
        "Secure Channel (Optimal State)",
        "Everything is operating perfectly. No intruders detected. The noise is just natural physics. This is our 'Baseline' or Golden Standard.",
        "Status: Monitoring raw quantum states.",
        "Conclusion: System is safe for top-secret data."
    ),
    "intercept": (
        "The Signal Tapper (Intercept-Resend)",
        "An attacker (Eve) is trying to 'listen' to the photons. Quantum physics says: 'You cannot look without changing'. Because Eve looked, the signal got 'smudged', causing an Error (QBER) spike.",
        "Status: Photons measured by unauthorized third-party.",
        "Conclusion: High Error Spike = Someone is tapping the line!"
    ),
    "pns": (
        "The Spare-Part Thief (PNS Attack)",
        "Our laser occasionally sends 2 photons instead of 1. Eve steals the 'extra' one quietly. Errors don't go up, but our Data Volume (Key) drops because Eve kept the spares.",
        "Status: Multi-photon pulses split and redirected.",
        "Conclusion: Key Length drop + Stable Error = Stealthy theft."
    ),
    "trojan": (
        "The Hardware Spy (Trojan Horse)",
        "Eve shines her own laser INTO our machine to see our settings via reflections. It is like a spy camera looking over our shoulder. This makes the data slightly less random.",
        "Status: Back-reflection analysis detected.",
        "Conclusion: Entropy drop suggests hardware is being spied on."
    ),
    "blinding": (
        "The Sensor Sabotage (Blinding)",
        "Eve hits our sensors with a massive flash of light. The sensors go 'blind' and stop working as quantum tools. Now Eve controls what the sensors report to us. Data loss is massive.",
        "Status: Detectors forced into classical saturation.",
        "Conclusion: Catastrophic Data Loss = Sensor Sabotage."
    ),
    "rng": (
        "The Loaded Dice (Weak RNG)",
        "The machine making our random numbers is biased—like a coin that always lands on 'Heads'. Eve can now predict our secrets. Bias goes up and Entropy goes down.",
        "Status: Predictable bit generation detected.",
        "Conclusion: High Bias = Security keys are easy to guess."
    ),
    "wavelength": (
        "The Color Trick (Wavelength Attack)",
        "Eve changes the 'color' (wavelength) of the light. Our detectors get confused and report wrong data patterns, skewing the balance (Bias) without increasing the error rate.",
        "Status: Spectrum manipulation detected.",
        "Conclusion: Bias shift detected without Error increase."
    ),
    "combined": (
        "Total System Breach (Combined Attack)",
        "Multiple attack methods are happening at once. Eve is tapping, stealing, and blinding the sensors simultaneously. The system shows failure in every single metric.",
        "Status: Multi-vector intrusion confirmed.",
        "Conclusion: TOTAL BREACH. Immediate shutdown recommended."
    )
}

# =========================================================
# MAIN GUI ARCHITECTURE
# =========================================================
class QKDApp:
    def __init__(self, root):
        self.root = root
        self.root.title("QKD ATTACK INTELLIGENCE DASHBOARD v6.0")
        self.root.state('zoomed') # Full screen fit
        self.root.configure(bg=CLR_BG)
        
        self.setup_styles()
        self.create_widgets()
        
    def setup_styles(self):
        style = ttk.Style()
        style.theme_use('clam')
        style.configure("TCombobox", fieldbackground=CLR_PANEL, background=CLR_ACCENT, 
                        foreground="white", font=("Consolas", 12))

    def create_widgets(self):
        # 1. Main Header
        header_frame = tk.Frame(self.root, bg=CLR_BG, pady=15)
        header_frame.pack(fill="x")
        tk.Label(header_frame, text="QUANTUM THREAT INTELLIGENCE", 
                 font=("Consolas", 28, "bold"), fg=CLR_ACCENT, bg=CLR_BG).pack()

        # 2. Split Layout using PanedWindow
        self.paned_window = tk.PanedWindow(self.root, orient=tk.HORIZONTAL, bg=CLR_BG, sashwidth=6)
        self.paned_window.pack(expand=True, fill="both", padx=15, pady=5)

        # --- LEFT SIDE: INFO & SELECTION ---
        left_pane = tk.Frame(self.paned_window, bg=CLR_BG, width=550)
        self.paned_window.add(left_pane)

        # Attack Selection Card
        sel_card = tk.LabelFrame(left_pane, text=" SELECT ATTACK VECTOR (Intrusion Method) ", 
                                 bg=CLR_PANEL, fg=CLR_ACCENT, font=("Consolas", 12, "bold"))
        sel_card.pack(fill="x", pady=5)
        self.atk_var = tk.StringVar(value=ATTACKS[0])
        self.combo = ttk.Combobox(sel_card, textvariable=self.atk_var, values=ATTACKS, 
                                  font=("Consolas", 14), state="readonly")
        self.combo.pack(fill="x", padx=15, pady=15)
        self.combo.bind("<<ComboboxSelected>>", self.update_info)

        # Scrollable Intelligence Panel
        intel_card = tk.LabelFrame(left_pane, text=" SYSTEM INTELLIGENCE GUIDE ", 
                                   bg=CLR_PANEL, fg=CLR_SUCCESS, font=("Consolas", 12, "bold"))
        intel_card.pack(expand=True, fill="both", pady=5)
        
        # Text widget for scrollable info
        self.info_text = tk.Text(intel_card, bg=CLR_PANEL, fg=CLR_TEXT, font=("Consolas", 11), 
                                 wrap="word", borderwidth=0, padx=15, pady=15)
        self.info_text.pack(side="left", expand=True, fill="both")
        
        scrollbar = tk.Scrollbar(intel_card, command=self.info_text.yview)
        scrollbar.pack(side="right", fill="y")
        self.info_text.config(yscrollcommand=scrollbar.set)

        # --- RIGHT SIDE: DATA VISUALIZATION ---
        right_pane = tk.Frame(self.paned_window, bg=CLR_BG)
        self.paned_window.add(right_pane)

        # Integrated Matplotlib Figure
        self.fig, self.ax = plt.subplots(figsize=(8, 6), facecolor=CLR_PANEL)
        self.ax.set_facecolor(CLR_PANEL)
        self.canvas = FigureCanvasTkAgg(self.fig, master=right_pane)
        self.canvas_widget = self.canvas.get_tk_widget()
        self.canvas_widget.pack(expand=True, fill="both", pady=5)

        # Full Comparison Result Box
        self.res_var = tk.StringVar(value="Waiting for signal analysis...")
        res_display = tk.Label(right_pane, textvariable=self.res_var, font=("Consolas", 12), 
                               fg=CLR_ACCENT, bg=CLR_PANEL, justify="left", 
                               padx=20, pady=20, relief="ridge", borderwidth=1)
        res_display.pack(fill="x", pady=5)

        # 3. Footer Action Button
        self.run_btn = tk.Button(self.root, text="EXECUTE QUANTUM ANALYSIS & DATASET COMPARISON", 
                                 command=self.run, bg=CLR_ACCENT, fg="black", 
                                 font=("Consolas", 18, "bold"), height=2)
        self.run_btn.pack(fill="x", padx=15, pady=15)

        # Initialize UI with Normal State
        self.update_info()
        self.run()

    def update_info(self, event=None):
        atk = self.atk_var.get()
        title, desc, logic, meaning = attack_db[atk]
        
        baseline_explanation = (
            "\n" + "="*45 + 
            "\n[1. WHERE DID THE BASELINE COME FROM?]\n"
            "This 'Blue Bar' represents the system's 'Golden State'. It was recorded "
            "during a Pre-Launch Calibration where no attacker was present. "
            "It captures the hardware's natural noise baseline."
            "\n\n[2. WHAT DOES 'DELTA' MEAN?]\n"
            "Delta is the 'Quantum Variance'. It is the mathematical distance "
            "between system safety and the current live signal. A high Delta "
            "indicates the laws of physics are being pushed by an intruder."
        )
        
        full_info = (
            f"{metric_guide}\n"
            f"{'='*45}\n"
            f"SELECTED VECTOR: {title.upper()}\n\n"
            f"[STORY & ANALYSIS]\n{desc}\n"
            f"{baseline_explanation}\n\n"
            f"[TECHNICAL RESULT]\n{meaning}"
        )
        
        self.info_text.config(state="normal")
        self.info_text.delete('1.0', tk.END)
        self.info_text.insert(tk.END, full_info)
        self.info_text.config(state="disabled")

    def graph(self, atk, q0, k0, b0, e0, l0, q, k, b, e, l):
        self.ax.clear()
        self.ax.set_facecolor(CLR_PANEL)
        
        labels = ["QBER", "KEYLEN", "BIAS", "ENTROPY", "LOSS"]
        base = [q0, k0, b0, e0, l0]
        attack = [q, k, b, e, l]
        x = [0, 1, 2, 3, 4]

        # Draw Comparison Bars
        b1 = self.ax.bar([i + 0.2 for i in x], base, width=0.4, label="Normal Baseline (Calibration)", color="#0055ff")
        b2 = self.ax.bar([i - 0.2 for i in x], attack, width=0.4, label="Live Attack Signal", color=CLR_WARNING)

        # SECURITY THRESHOLD LINE
        threshold_val = 25 
        self.ax.axhline(y=threshold_val, color=CLR_WARNING, linestyle='--', alpha=0.6, label="Security Threshold")

        # --- DYNAMIC HIGHLIGHTING & DIRECTIONAL LOGIC ---
        highlight_map = {
            "intercept": [0], "pns": [1], "blinding": [1, 4], 
            "rng": [2, 3], "wavelength": [2], "combined": [0, 1, 2, 3, 4]
        }
        
        relevant_indices = highlight_map.get(atk, [])
        for idx in relevant_indices:
            self.ax.axvspan(idx-0.4, idx+0.4, color=CLR_ACCENT, alpha=0.15)
            
            # DIRECTIONAL LOGIC (Increased or Decreased)
            direction = "↑ INCREASE" if attack[idx] > base[idx] else "↓ DECREASE"
            
            self.ax.text(idx, max(base[idx], attack[idx]) + 5, f"ANOMALY\n{direction}", 
                         color=CLR_ACCENT, ha='center', fontsize=12, fontweight='bold')

        # Axis and Labels Styling
        self.ax.set_xticks(x)
        self.ax.set_xticklabels(labels, color=CLR_TEXT, fontweight='bold', fontsize=12)
        self.ax.set_ylabel("Quantity Level / Scalar", color=CLR_ACCENT, fontsize=14)
        self.ax.tick_params(colors=CLR_TEXT)
        self.ax.set_title(f"LIVE DATASET ANALYSIS: {atk.upper()}", color=CLR_ACCENT, pad=25, fontsize=18)

        # Numerical Values above bars
        for bar_set in [b1, b2]:
            for rect in bar_set:
                yval = rect.get_height()
                self.ax.text(rect.get_x() + rect.get_width()/2, yval + 1, f'{yval:.2f}', 
                             ha='center', va='bottom', color=CLR_TEXT, fontsize=10, fontweight='bold')

        self.ax.legend(facecolor=CLR_PANEL, labelcolor=CLR_TEXT, fontsize=12)
        self.canvas.draw()

    def run(self):
        atk = self.atk_var.get()
        # Fetching data from simulation logic
        q0, k0, b0, e0, l0 = simulate("normal") # Baseline data
        q, k, b, e, l = simulate(atk)           # Live data based on selection
        
        # Detailed results text panel
        self.res_var.set(
            f"DATASET BASELINE (ORIGINAL) | QBER: {q0:.3f} | KEY: {k0} | BIAS: {b0:.3f} | LOSS: {l0:.3f}\n"
            f"LIVE SIGNAL RESULT (CURRENT) | QBER: {q:.3f} | KEY: {k} | BIAS: {b:.3f} | LOSS: {l:.3f}\n"
            f"SYSTEM VARIANCE (DELTA)      | ΔQBER: {q-q0:+.3f} | ΔKEY: {k-k0} | ΔBIAS: {b-b0:+.3f} | ΔLOSS: {l-l0:+.3f}"
        )
        
        self.graph(atk, q0, k0, b0, e0, l0, q, k, b, e, l)

# =========================================================
# STARTING THE APPLICATION
# =========================================================
if __name__ == "__main__":
    root = tk.Tk()
    app = QKDApp(root)
    root.mainloop()