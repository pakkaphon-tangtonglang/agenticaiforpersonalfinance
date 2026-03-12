"""Constants and CSS for the Streamlit app."""

INTENT_CONFIG: dict[str, dict[str, str]] = {
    "tax": {"label": "ภาษี", "icon": "🧾", "color": "#FF6B6B"},
    "expense": {"label": "ค่าใช้จ่าย", "icon": "💸", "color": "#FFA94D"},
    "investment": {"label": "การลงทุน", "icon": "📈", "color": "#51CF66"},
    "planning": {"label": "วางแผนการเงิน", "icon": "🗓️", "color": "#339AF0"},
    "recommendation": {"label": "คำแนะนำการเงิน", "icon": "💡", "color": "#CC5DE8"},
    "report": {"label": "รายงานการเงิน", "icon": "📊", "color": "#20C997"},
    "general": {"label": "ทั่วไป", "icon": "💬", "color": "#868E96"},
    "unknown": {"label": "ไม่ทราบ", "icon": "❓", "color": "#868E96"},
    "error": {"label": "ข้อผิดพลาด", "icon": "⚠️", "color": "#FA5252"},
}

SAMPLE_QUERIES: list[dict[str, str]] = [
    {
        "short": "🧾 คำนวณภาษี",
        "full": "คำนวณภาษี เงินเดือน 50,000 บาท/เดือน มีลูก 1 คน ซื้อ SSF 100,000",
    },
    {"short": "💸 บันทึกค่าอาหาร", "full": "จ่ายค่าอาหาร 350 บาท"},
    {"short": "📋 สรุปรายจ่ายเดือนนี้", "full": "สรุปรายจ่ายเดือนนี้"},
    {"short": "📈 เพิ่มหุ้น PTT", "full": "เพิ่มหุ้น PTT.BK 100 หุ้น ราคา 35 บาท"},
    {"short": "💼 ดูพอร์ตการลงทุน", "full": "ดูพอร์ตของฉัน"},
]

MULTI_AGENT_QUERIES: list[dict[str, str]] = [
    {
        "short": "คำนวณภาษี + ดูพอร์ตลงทุน",
        "full": (
            "ช่วยคำนวณภาษีปี 2026 ให้หน่อย เงินเดือนเดือนละ 60,000 บาท"
            " มีลูก 1 คน ซื้อ SSF 100,000 บาท"
            " แล้วช่วยดูพอร์ตการลงทุนของฉันด้วยว่ามีกำไรขาดทุนเท่าไหร่"
            " อยากรู้ว่ามีผลกระทบทางภาษีไหม"
        ),
    },
    {
        "short": "วางแผนการเงิน + ดูรายจ่าย",
        "full": (
            "อยากวางแผนการเงินปีนี้ ช่วยดูเป้าหมายทั้งหมดของฉันให้หน่อย"
            " แล้วดึงข้อมูลรายจ่ายเดือนนี้กับรายได้ปีนี้มาเทียบด้วย"
            " ว่าฉันออมเงินได้ตามเป้าไหม ต้องปรับอะไรบ้าง"
        ),
    },
    {
        "short": "สรุปค่าใช้จ่าย + เป้าหมายการออม",
        "full": (
            "สรุปค่าใช้จ่ายเดือนนี้ให้หน่อย แยกตามหมวดหมู่"
            " แล้วดูเป้าหมายการเงินของฉันด้วยว่าค่าใช้จ่ายที่เป็นอยู่"
            " กระทบกับเป้าหมายการออมไหม ถ้ากระทบแนะนำวิธีลดรายจ่ายด้วย"
        ),
    },
]

REPORT_QUERIES: list[dict[str, str]] = [
    {
        "short": "รายงานการเงินประจำเดือน",
        "full": (
            "สร้างรายงานการเงินประจำเดือนนี้ให้หน่อย"
            " อยากเห็นภาพรวมรายรับรายจ่าย พอร์ตการลงทุน"
            " เป้าหมายการเงิน และสถานะภาษีทั้งหมด"
        ),
    },
    {
        "short": "รายงานสุขภาพการเงิน",
        "full": ("ขอดูรายงานสรุปสุขภาพการเงินของฉัน" " พร้อมคะแนนสุขภาพการเงินและไฮไลท์สำคัญ"),
    },
]

RECOMMENDATION_QUERIES: list[dict[str, str]] = [
    {
        "short": "วิเคราะห์การเงินรวม + คำแนะนำ",
        "full": (
            "ช่วยวิเคราะห์การเงินทั้งหมดของฉันให้หน่อย"
            " ดูรายจ่าย รายได้ ภาษี พอร์ตการลงทุน และเป้าหมายทุกอย่าง"
            " แล้วให้คำแนะนำเชิงรุกว่าควรปรับปรุงจุดไหนบ้าง"
            " เรียงตามความเร่งด่วนจากมากไปน้อย"
        ),
    },
    {
        "short": "ตรวจสุขภาพการเงิน + คะแนน",
        "full": (
            "ตรวจสุขภาพการเงินของฉันให้หน่อย"
            " อยากรู้คะแนนสุขภาพการเงินของฉัน"
            " และปัญหาเร่งด่วนที่ต้องแก้ไขก่อน 3 อันดับแรก"
            " พร้อมแนะนำว่าต้องทำอะไรบ้างเพื่อเพิ่มคะแนน"
        ),
    },
]

FEATURE_CARDS: list[dict[str, str]] = [
    {"icon": "🧾", "title": "คำนวณภาษี", "desc": "วางแผนภาษีเงินได้\nหักลดหย่อนอัตโนมัติ"},
    {"icon": "💸", "title": "จัดการค่าใช้จ่าย", "desc": "บันทึกรายจ่าย\nวิเคราะห์แยกหมวดหมู่"},
    {"icon": "📈", "title": "ติดตามการลงทุน", "desc": "ดูพอร์ต กำไร/ขาดทุน\nหุ้น กองทุนรวม"},
    {"icon": "🗓️", "title": "วางแผนการเงิน", "desc": "ตั้งเป้าหมาย\nติดตามความคืบหน้า"},
    {"icon": "💡", "title": "คำแนะนำอัจฉริยะ", "desc": "วิเคราะห์เชิงรุก\nแนะนำปรับปรุง"},
    {"icon": "📊", "title": "รายงานสรุป", "desc": "สุขภาพการเงิน\nภาพรวมครบถ้วน"},
]

CUSTOM_CSS = """
<style>
    .main-header {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        padding: 1.5rem 2rem; border-radius: 16px;
        margin-bottom: 1.5rem; color: white;
    }
    .main-header h1 { margin: 0; font-size: 2rem; font-weight: 700; color: white; }
    .main-header p { margin: 0.3rem 0 0 0; opacity: 0.9; font-size: 0.95rem; color: white; }
    .intent-badge {
        display: inline-block; padding: 0.2rem 0.7rem; border-radius: 20px;
        font-size: 0.78rem; font-weight: 600; margin-bottom: 0.5rem; color: white;
    }
    section[data-testid="stSidebar"] .stButton > button {
        border-radius: 8px; font-size: 0.85rem;
        border: 1px solid rgba(128, 128, 128, 0.3); transition: all 0.2s;
    }
    section[data-testid="stSidebar"] .stButton > button:hover {
        border-color: #667eea; color: #839dff;
    }
    .new-chat-btn > button {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%) !important;
        color: white !important; font-weight: 600 !important;
        border: none !important; border-radius: 10px !important;
    }
    .new-chat-btn > button:hover { opacity: 0.9; color: white !important; }
    .conv-active > button {
        background: rgba(102, 126, 234, 0.15) !important;
        border-color: #667eea !important; color: #839dff !important;
        font-weight: 600 !important;
    }
    .stChatMessage { border-radius: 12px !important; margin-bottom: 0.5rem; }
    .clear-btn > button {
        color: #ff6b6b !important; border-color: rgba(255, 107, 107, 0.3) !important;
    }
    .clear-btn > button:hover {
        background: rgba(255, 107, 107, 0.1) !important;
        border-color: #ff6b6b !important;
    }
</style>
"""
