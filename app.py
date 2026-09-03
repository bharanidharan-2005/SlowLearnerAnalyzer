import os
import re
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import io
import base64
import sqlite3  
import json     
from flask import Flask, render_template, request, send_file, redirect, url_for, session, flash, jsonify

# --- Import our RAG AI pipeline ---
from rag_service import build_vector_db, ask_ai

app = Flask(__name__)
app.secret_key = 'mount_zion_secret_key'

# --- Configuration ---
UPLOAD_FOLDER = 'uploads'
REPORT_FOLDER = 'reports'
STATIC_FOLDER = 'static'
PASS_MARK = 50

os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(REPORT_FOLDER, exist_ok=True)
os.makedirs(STATIC_FOLDER, exist_ok=True)

# --- DATABASE SETUP ---
DB_NAME = "instance/students.db"
os.makedirs("instance", exist_ok=True)

def init_db():
    conn = sqlite3.connect(DB_NAME)
    c = conn.cursor()
    c.execute('''
        CREATE TABLE IF NOT EXISTS student_performance (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            exam_name TEXT,
            register_no TEXT,
            student_name TEXT,
            marks_json TEXT,
            failed_subjects TEXT,
            total INTEGER,
            average REAL,
            status TEXT
        )
    ''')
    conn.commit()
    conn.close()

init_db()

USERS = {
    "admin": "admin123",
    "faculty": "pass",
    "hod": "secure"
}

def generate_student_graph(marks_dict):
    fig, ax = plt.subplots(figsize=(5, 2.5))
    subjects = list(marks_dict.keys())
    scores = []
    colors = []
    
    for sub in subjects:
        val = str(marks_dict[sub]).strip().upper()
        if val in ['AB', 'A', 'ABS', 'ABSENT']:
            scores.append(0)
            colors.append('#e74c3c') 
        else:
            try:
                mark = float(val)
                scores.append(mark)
                colors.append('#2ecc71' if mark >= PASS_MARK else '#e74c3c') 
            except:
                scores.append(0)
                colors.append('#bdc3c7')

    ax.bar(subjects, scores, color=colors)
    ax.set_ylim(0, 100)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    plt.xticks(rotation=15, fontsize=9)
    plt.tight_layout()
    
    buf = io.BytesIO()
    plt.savefig(buf, format='png', transparent=True)
    plt.close(fig)
    buf.seek(0)
    return base64.b64encode(buf.read()).decode('utf-8')

def extract_all_students(filepath, filename):
    try:
        if filepath.endswith('.csv'):
            try: df = pd.read_csv(filepath, header=None, encoding='utf-8')
            except: df = pd.read_csv(filepath, header=None, encoding='ISO-8859-1')
        else:
            df = pd.read_excel(filepath, header=None)
    except Exception as e:
        return None, f"File read error: {str(e)}"

    reg_col, name_col = -1, -1
    for r in range(min(20, len(df))):
        for c in range(df.shape[1]):
            val = str(df.iloc[r, c]).strip().upper()
            if 'REG' in val and ('NO' in val or 'NUM' in val): reg_col = c
            if val == 'NAME' or 'STUDENT' in val: name_col = c

    if reg_col == -1: reg_col = 2
    if name_col == -1: name_col = 3

    data_start_row = -1
    for r in range(min(20, len(df))):
        val = str(df.iloc[r, reg_col]).strip()
        if len(val) >= 6 and any(char.isdigit() for char in val) and "REG" not in val.upper():
            data_start_row = r
            break

    if data_start_row == -1: return None, "Could not identify student data."

    subj_row = data_start_row - 1
    subjects = {}
    for c in range(name_col + 1, df.shape[1]):
        val = str(df.iloc[subj_row, c]).strip()
        is_date = bool(re.search(r'\d{1,2}[\.\-/]\d{1,2}', val))
        
        if not val or val.lower() == 'nan' or is_date:
            if subj_row - 1 >= 0:
                val_up = str(df.iloc[subj_row - 1, c]).strip()
                if val_up and val_up.lower() != 'nan' and not bool(re.search(r'\d{1,2}[\.\-/]\d{1,2}', val_up)):
                    val = val_up
                    
        if val and val.lower() != 'nan' and not is_date and 'total' not in val.lower() and 'avg' not in val.lower():
            subjects[c] = val

    if not subjects: return None, "Could not extract subjects."

    all_students = []
    exam_name = filename.split('.')[0]

    for r in range(data_start_row, len(df)):
        reg = str(df.iloc[r, reg_col]).strip().replace('.0', '')
        name = str(df.iloc[r, name_col]).strip()

        if not reg or reg.lower() == 'nan' or not name or name.lower() in ['nan', 'pass', 'fail', 'absent', 'total']: 
            continue

        marks_dict = {}
        failures = []
        total_marks = 0
        valid_subs = 0

        for c, subj in subjects.items():
            mark_str = str(df.iloc[r, c]).strip().upper()
            marks_dict[subj] = mark_str

            if mark_str in ['AB', 'A', 'ABS', 'ABSENT']:
                failures.append(f"{subj} (AB)")
            elif mark_str not in ['NAN', '', 'NA']:
                try:
                    mark = float(mark_str)
                    total_marks += mark
                    valid_subs += 1
                    if mark < PASS_MARK: failures.append(f"{subj} ({int(mark)})")
                except: pass

        avg = round(total_marks / valid_subs, 2) if valid_subs > 0 else 0

        all_students.append({
            "Exam": exam_name,
            "Register No": reg,
            "Student Name": name,
            "Marks": marks_dict,
            "Failed Subjects": ", ".join(failures) if failures else "None",
            "Total": total_marks,
            "Average": avg,
            "Status": "Pass" if not failures else "Fail"
        })

    return all_students, None

# ================= ROUTES =================

@app.route("/")
def home():
    return render_template("login.html")

@app.route('/login', methods=['GET', 'POST'])
def login():
    if request.method == 'POST':
        if request.form['username'] in USERS and USERS[request.form['username']] == request.form['password']:
            session['user'] = request.form['username']
            return redirect(url_for('dashboard'))
        flash("Invalid credentials", "danger")
    return render_template('login.html')

@app.route('/logout')
def logout():
    session.pop('user', None)
    return redirect(url_for('login'))

@app.route('/dashboard', methods=['GET', 'POST'])
def dashboard():
    if 'user' not in session: return redirect(url_for('login'))

    data = None

    if request.method == 'POST':
        files = request.files.getlist('files')
        if not files or files[0].filename == '':
            return render_template('dashboard.html', error="No files selected.")

        all_students = []
        for file in files:
            if file.filename == '': continue
            filepath = os.path.join(UPLOAD_FOLDER, file.filename)
            file.save(filepath)
            
            students, msg = extract_all_students(filepath, file.filename)
            if students: all_students.extend(students)

        if not all_students:
            return render_template('dashboard.html', error="Could not process data. Check file format.")

        # --- SAVE EXTRACTED DATA TO SQLITE DATABASE ---
        conn = sqlite3.connect(DB_NAME)
        c = conn.cursor()
        c.execute("DELETE FROM student_performance") # Clear old session data
        
        for s in all_students:
            c.execute('''
                INSERT INTO student_performance 
                (exam_name, register_no, student_name, marks_json, failed_subjects, total, average, status)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?)
            ''', (
                s['Exam'], 
                s['Register No'], 
                s['Student Name'], 
                json.dumps(s['Marks']), 
                s['Failed Subjects'], 
                s['Total'], 
                s['Average'], 
                s['Status']
            ))
        conn.commit()
        conn.close()
        
        # --- TRIGGER RAG PIPELINE ---
        print("Starting RAG Ingestion Pipeline...")
        build_vector_db()
        # ---------------------------------

        failures = [s for s in all_students if s['Status'] == 'Fail']
        passes = [s for s in all_students if s['Status'] == 'Pass']
        toppers = sorted(passes, key=lambda x: x['Total'], reverse=True)[:15] 

        for s in all_students:
            s['Graph_Base64'] = generate_student_graph(s['Marks'])
            
        total_students = len(all_students)
        pass_pct = round((len(passes) / total_students) * 100, 1) if total_students > 0 else 0
        top_score = toppers[0]['Total'] if toppers else 0
        
        subj_fails = {}
        for s in failures:
            if s['Failed Subjects'] != "None":
                for fail_item in s['Failed Subjects'].split(', '):
                    sub_name = fail_item.split(' (')[0]
                    subj_fails[sub_name] = subj_fails.get(sub_name, 0) + 1

        stats = {
            "total_students": total_students,
            "pass_pct": pass_pct,
            "fail_count": len(failures),
            "top_score": top_score,
            "subj_fails_keys": list(subj_fails.keys()),
            "subj_fails_values": list(subj_fails.values())
        }

        with pd.ExcelWriter(os.path.join(REPORT_FOLDER, 'Comprehensive_Report.xlsx')) as writer:
            if failures:
                pd.DataFrame([{k: v for k, v in s.items() if k not in ['Marks', 'Graph_Base64']} for s in failures]).to_excel(writer, sheet_name='Slow Learners', index=False)
            if toppers:
                pd.DataFrame([{k: v for k, v in s.items() if k not in ['Marks', 'Graph_Base64']} for s in toppers]).to_excel(writer, sheet_name='High Achievers', index=False)
            pd.DataFrame([{k: v for k, v in s.items() if k not in ['Marks', 'Graph_Base64']} for s in all_students]).to_excel(writer, sheet_name='All Students', index=False)

        data = {
            "toppers": toppers,
            "failures": failures,
            "all_students": all_students,
            "stats": stats
        }

    return render_template('dashboard.html', data=data)

@app.route('/download')
def download():
    if 'user' not in session: return redirect(url_for('login'))
    return send_file(os.path.join(REPORT_FOLDER, 'Comprehensive_Report.xlsx'), as_attachment=True)

# ==========================================
# NEW: AI CHAT API ROUTE
# ==========================================
@app.route('/api/chat', methods=['POST'])
def chat():
    # Make sure only logged-in faculty can use the AI
    if 'user' not in session: 
        return jsonify({"error": "Unauthorized"}), 401
        
    data = request.get_json()
    question = data.get("question")
    
    if not question:
        return jsonify({"error": "No question provided"}), 400

    # Call our RAG pipeline from rag_service.py!
    answer = ask_ai(question)
    
    return jsonify({"answer": answer})
# ==========================================

if __name__ == '__main__':
    app.run(debug=True)