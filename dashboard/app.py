"""
Streamlit dashboard for the Attendance Bot.

Run with: streamlit run dashboard/app.py
Make sure the FastAPI server is running first: uvicorn app.main:app --reload
"""
import requests
import streamlit as st

API_BASE = "http://localhost:8000"

st.set_page_config(page_title="Attendance Bot", layout="wide")
st.title("Attendance Bot Dashboard")

page = st.sidebar.radio("Navigate", ["Register Student", "Start Meeting", "View Attendance"])

# ── Page 1: Register Student ────────────────────────────────────────
if page == "Register Student":
    st.header("Register a New Student")

    with st.form("register_form"):
        name = st.text_input("Full Name")
        student_id = st.text_input("Student ID")
        photo = st.file_uploader("Upload Selfie", type=["jpg", "jpeg", "png"])
        submitted = st.form_submit_button("Register")

    if submitted:
        if not name or not student_id or not photo:
            st.error("Please fill in all fields and upload a photo.")
        else:
            try:
                response = requests.post(
                    f"{API_BASE}/register/",
                    data={"name": name, "student_id": student_id},
                    files={"photo": (photo.name, photo.getvalue(), photo.type)},
                )
                if response.status_code == 200:
                    st.success(response.json()["message"])
                else:
                    st.error(response.json().get("detail", "Registration failed."))
            except requests.ConnectionError:
                st.error("Cannot connect to the API server. Is it running?")

    # Show registered students
    st.subheader("Registered Students")
    try:
        response = requests.get(f"{API_BASE}/register/students")
        if response.status_code == 200:
            students = response.json()
            if students:
                st.table(students)
            else:
                st.info("No students registered yet.")
    except requests.ConnectionError:
        st.warning("Cannot connect to the API server.")


# ── Page 2: Start Meeting ───────────────────────────────────────────
elif page == "Start Meeting":
    st.header("Start Attendance Session")

    st.info(
        "**Stub mode**: Enter a folder path containing test images (JPG/PNG) "
        "to simulate a meeting.\n\n"
        "**Production mode**: Enter a Zoom, Teams, or Google Meet URL."
    )

    meeting_url = st.text_input("Meeting URL or Test Folder Path")

    if st.button("Start Session"):
        if not meeting_url:
            st.error("Please enter a meeting URL or folder path.")
        else:
            try:
                response = requests.post(
                    f"{API_BASE}/meeting/start",
                    json={"meeting_url": meeting_url},
                )
                if response.status_code == 200:
                    result = response.json()
                    st.success(f"Session #{result['session_id']} started!")
                    st.json(result)
                else:
                    st.error(response.json().get("detail", "Failed to start session."))
            except requests.ConnectionError:
                st.error("Cannot connect to the API server. Is it running?")


# ── Page 3: View Attendance ─────────────────────────────────────────
elif page == "View Attendance":
    st.header("Attendance Reports")

    # List all sessions
    try:
        response = requests.get(f"{API_BASE}/attendance/")
        if response.status_code == 200:
            sessions = response.json()
            if not sessions:
                st.info("No attendance sessions yet. Start a meeting first.")
            else:
                session_options = {
                    f"Session #{s['id']} — {s['started_at'][:16]}": s["id"]
                    for s in sessions
                }
                selected = st.selectbox("Select a session", list(session_options.keys()))
                session_id = session_options[selected]

                # Fetch and display the report
                report_response = requests.get(f"{API_BASE}/attendance/{session_id}")
                if report_response.status_code == 200:
                    data = report_response.json()
                    report = data["report"]

                    if report:
                        # Summary stats
                        present = sum(1 for r in report if r["status"] == "Present")
                        absent = sum(1 for r in report if r["status"] == "Absent")
                        col1, col2, col3 = st.columns(3)
                        col1.metric("Present", present)
                        col2.metric("Absent", absent)
                        col3.metric("Total", len(report))

                        # Detailed table
                        st.table(report)

                        # CSV download
                        csv_response = requests.get(f"{API_BASE}/attendance/{session_id}/export")
                        if csv_response.status_code == 200:
                            st.download_button(
                                label="Download CSV",
                                data=csv_response.text,
                                file_name=f"attendance_session_{session_id}.csv",
                                mime="text/csv",
                            )
                    else:
                        st.info("No attendance data for this session.")
    except requests.ConnectionError:
        st.warning("Cannot connect to the API server.")
