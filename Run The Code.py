import os
import sys
import threading
import subprocess
import webbrowser
import tkinter as tk
from tkinter import ttk, messagebox, filedialog
from tkinter.scrolledtext import ScrolledText

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
SCRIPTS_DIR = os.path.join(BASE_DIR, 'scripts')
OUTPUTS_DIR = os.path.join(BASE_DIR, 'outputs')
POWERBI_DIR = os.path.join(BASE_DIR, 'powerbi')
NOTEBOOKS_DIR = os.path.join(BASE_DIR, 'notebooks')
os.makedirs(POWERBI_DIR, exist_ok=True)
os.makedirs(NOTEBOOKS_DIR, exist_ok=True)

PBIX_NAME = "تحليل بيانات تأثير الكاء الاصطناعي على المجتمع.pbix"
IPYNB_NAME = "تحليل_بيانات_تأثير_الذكاء_الاصطناعي_على_المجتمع.ipynb"

def _python_cmd():
    for cmd in ['py', 'python', 'python3']:
        try:
            subprocess.run([cmd, '--version'], stdout=subprocess.PIPE, stderr=subprocess.PIPE, check=True)
            return cmd
        except Exception:
            continue
    return None

def _log_msg(log_widget, text, tag=None):
    log_widget.configure(state='normal')
    if tag:
        log_widget.insert('end', text + "\n", tag)
    else:
        log_widget.insert('end', text + "\n")
    log_widget.see('end')
    log_widget.configure(state='disabled')

def run_script(script_name, log_widget, friendly_name=None):
    cmd = _python_cmd()
    if not cmd:
        messagebox.showerror("Error", "Python is not available on this system.")
        return
    script_path = os.path.join(SCRIPTS_DIR, script_name)
    if not os.path.exists(script_path):
        messagebox.showerror("Error", f"Script not found: {script_path}")
        return
    def _runner():
        title = friendly_name or script_name
        _log_msg(log_widget, f"▶ Starting {title} ...", 'info')
        try:
            proc = subprocess.Popen([cmd, script_path], cwd=BASE_DIR, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
            for line in proc.stdout:
                # stream output in neutral color
                _log_msg(log_widget, line.rstrip(), None)
            proc.wait()
            code = proc.returncode
            if code == 0:
                _log_msg(log_widget, f"✓ {title} completed successfully", 'success')
            else:
                _log_msg(log_widget, f"✗ {title} failed (exit code {code})", 'error')
        except Exception as e:
            _log_msg(log_widget, f"Error: {e}", 'error')
    threading.Thread(target=_runner, daemon=True).start()

def list_files(dir_path, extensions):
    files = []
    if os.path.exists(dir_path):
        for root, _, names in os.walk(dir_path):
            for n in names:
                if any(n.lower().endswith(ext) for ext in extensions):
                    files.append(os.path.join(root, n))
    return sorted(files)

def open_file(path):
    try:
        ext = os.path.splitext(path)[1].lower()
        if ext == '.html':
            webbrowser.open('file://' + os.path.abspath(path))
        else:
            if sys.platform == 'win32':
                os.startfile(path)
            elif sys.platform == 'darwin':
                subprocess.call(['open', path])
            else:
                subprocess.call(['xdg-open', path])
    except Exception:
        pass

def show_files_window(title, dir_paths, exts):
    win = tk.Toplevel()
    win.title(title)
    win.geometry("900x520")
    frame = ttk.Frame(win)
    frame.pack(fill='both', expand=True, padx=10, pady=10)
    lb = tk.Listbox(frame)
    lb.pack(fill='both', expand=True)
    def refresh():
        lb.delete(0, 'end')
        all_files = []
        for d in dir_paths:
            all_files.extend(list_files(d, exts))
        if not all_files:
            lb.insert('end', 'No files found')
        else:
            for f in all_files:
                lb.insert('end', f)
    refresh()
    def on_open():
        sel = lb.curselection()
        if not sel:
            return
        path = lb.get(sel[0])
        if os.path.isfile(path):
            open_file(path)
    button_bar = ttk.Frame(frame)
    button_bar.pack(fill='x', pady=8)
    ttk.Button(button_bar, text="Refresh", command=refresh).pack(side='left')
    ttk.Button(button_bar, text="Open Selected", command=on_open).pack(side='left', padx=6)
    lb.bind('<Double-Button-1>', lambda e: on_open())

def open_fixed_pbix():
    candidates = [
        os.path.join(POWERBI_DIR, PBIX_NAME),
        os.path.join(BASE_DIR, PBIX_NAME)
    ]
    for p in candidates:
        if os.path.exists(p):
            open_file(p)
            return
    messagebox.showerror("Error", f"Power BI file not found:\n{PBIX_NAME}\nPlace it under: {POWERBI_DIR}")

def open_fixed_ipynb():
    candidates = [
        os.path.join(NOTEBOOKS_DIR, IPYNB_NAME),
        os.path.join(BASE_DIR, IPYNB_NAME)
    ]
    for p in candidates:
        if os.path.exists(p):
            open_file(p)
            return
    messagebox.showerror("Error", f"Notebook not found:\n{IPYNB_NAME}\nPlace it under: {NOTEBOOKS_DIR}")

def main():
    root = tk.Tk()
    root.title("AI Impact Study - Control Panel")
    root.geometry("1000x700")

    style = ttk.Style()
    try:
        style.theme_use('vista' if sys.platform == 'win32' else 'clam')
    except Exception:
        pass
    # Button styles
    style.configure('Primary.TButton', foreground='black', background='#1f77b4', padding=6)
    style.map('Primary.TButton', background=[('active', '#1663a6')])
    style.configure('Success.TButton', foreground='black', background='#2ca02c', padding=6)
    style.map('Success.TButton', background=[('active', '#208220')])
    style.configure('Info.TButton', foreground='black', background='#17a2b8', padding=6)
    style.map('Info.TButton', background=[('active', '#12879a')])
    style.configure('Warn.TButton', foreground='black', background='#ff7f0e', padding=6)
    style.map('Warn.TButton', background=[('active', '#e6760d')])

    notebook = ttk.Notebook(root)
    notebook.pack(fill='both', expand=True, padx=10, pady=10)

    tasks_tab = ttk.Frame(notebook)
    outputs_tab = ttk.Frame(notebook)
    notebook.add(tasks_tab, text="Tasks")
    notebook.add(outputs_tab, text="Outputs")

    # Tasks Tab
    top = ttk.LabelFrame(tasks_tab, text="Run Pipelines")
    top.pack(fill='x', padx=10, pady=10)

    global log
    log_frame = ttk.LabelFrame(tasks_tab, text="Execution Log")
    log_frame.pack(fill='both', expand=True, padx=10, pady=10)
    log = ScrolledText(log_frame, wrap='word', state='disabled', height=20)
    log.pack(fill='both', expand=True)
    # Log color tags
    log.tag_config('info', foreground='#1f77b4')
    log.tag_config('success', foreground='#2ca02c')
    log.tag_config('error', foreground='#d62728')
    log.tag_config('warn', foreground='#ff7f0e')

    btn_clean = ttk.Button(top, text="Run Data Cleaning", style='Primary.TButton', command=lambda: run_script('data_cleaning.py', log, 'Data Cleaning'))
    btn_stats = ttk.Button(top, text="Run Statistical Analysis", style='Primary.TButton', command=lambda: run_script('statistical_analysis.py', log, 'Statistical Analysis'))
    btn_models = ttk.Button(top, text="Run AI Models", style='Primary.TButton', command=lambda: run_script('predictive_models.py', log, 'AI Models'))
    btn_interactive_gen = ttk.Button(top, text="Generate Interactive Plots", style='Info.TButton', command=lambda: run_script('interactive_plots.py', log, 'Interactive Plots Generation'))

    btn_clean.grid(row=0, column=0, padx=5, pady=8, sticky='w')
    btn_stats.grid(row=0, column=1, padx=5, pady=8, sticky='w')
    btn_models.grid(row=0, column=2, padx=5, pady=8, sticky='w')
    btn_interactive_gen.grid(row=0, column=3, padx=5, pady=8, sticky='w')

    # Outputs Tab
    out_top = ttk.LabelFrame(outputs_tab, text="Browse Outputs and Assets")
    out_top.pack(fill='x', padx=10, pady=10)

    btn_interactive = ttk.Button(out_top, text="View Interactive Plots", style='Info.TButton', command=lambda: show_files_window("Interactive Plots", [os.path.join(OUTPUTS_DIR, '07_interactive_plots')], ['.html']))
    btn_static = ttk.Button(out_top, text="View Static Plots", style='Info.TButton', command=lambda: show_files_window("Static Plots", [os.path.join(OUTPUTS_DIR, '02_visualizations'), os.path.join(OUTPUTS_DIR, '05_clustering'), os.path.join(OUTPUTS_DIR, '06_predictive_models')], ['.png', '.jpg', '.jpeg']))
    btn_outputs = ttk.Button(out_top, text="Open Outputs Folder", style='Success.TButton', command=lambda: open_file(OUTPUTS_DIR))
    btn_powerbi_folder = ttk.Button(out_top, text="Open Power BI Folder", style='Success.TButton', command=lambda: open_file(POWERBI_DIR))
    btn_notebooks_folder = ttk.Button(out_top, text="Open Notebooks Folder", style='Success.TButton', command=lambda: open_file(NOTEBOOKS_DIR))

    def open_pbix():
        open_fixed_pbix()
    def open_ipynb():
        open_fixed_ipynb()

    btn_open_pbix = ttk.Button(out_top, text="Open Power BI (.pbix)", style='Warn.TButton', command=open_pbix)
    btn_open_ipynb = ttk.Button(out_top, text="Open Notebook (.ipynb)", style='Warn.TButton', command=open_ipynb)

    btn_interactive.grid(row=0, column=0, padx=5, pady=8, sticky='w')
    btn_static.grid(row=0, column=1, padx=5, pady=8, sticky='w')
    btn_outputs.grid(row=0, column=2, padx=5, pady=8, sticky='w')
    btn_powerbi_folder.grid(row=1, column=0, padx=5, pady=8, sticky='w')
    btn_notebooks_folder.grid(row=1, column=1, padx=5, pady=8, sticky='w')
    btn_open_pbix.grid(row=2, column=0, padx=5, pady=8, sticky='w')
    btn_open_ipynb.grid(row=2, column=1, padx=5, pady=8, sticky='w')

    # Status Bar
    status = ttk.Label(root, text=f"Base: {BASE_DIR}", anchor='w')
    status.pack(fill='x', padx=10, pady=5)

    root.mainloop()

if __name__ == '__main__':
    main()
