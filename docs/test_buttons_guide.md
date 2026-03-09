# คู่มือปุ่มทดสอบ Streamlit

## ภาพรวม

ปุ่มทดสอบใน sidebar ของ Streamlit UI แบ่งเป็น 2 กลุ่ม:

1. **ทดสอบ Multi-Agent** — ทดสอบความสามารถในการดึงข้อมูลข้ามโดเมน (Cross-Agent Collaboration)
2. **ทดสอบคำแนะนำเชิงรุก** — ทดสอบ Recommendation Agent ที่วิเคราะห์ข้อมูลทุกด้านแล้วให้คำแนะนำ

---

## 1. ทดสอบ Multi-Agent (Cross-Agent Collaboration)

### แนวคิด

ปกติแต่ละ Agent จัดการเฉพาะโดเมนของตัวเอง:
- **Tax Agent** → ภาษี
- **Expense Agent** → ค่าใช้จ่าย
- **Investment Agent** → การลงทุน
- **Planning Agent** → วางแผนการเงิน/เป้าหมาย

**Multi-Agent Collaboration** ทำให้ Agent ตัวหนึ่งสามารถดึงข้อมูลจากโดเมนอื่นได้ ผ่าน Cross-Agent Tools:

| Cross-Agent Tool | ข้อมูลที่ดึง | Agent ที่ใช้ได้ |
|---|---|---|
| `get_expense_summary_cross` | สรุปค่าใช้จ่ายรายเดือน | Tax, Planning |
| `get_portfolio_summary_cross` | สรุปพอร์ตการลงทุน | Tax, Planning |
| `get_goals_summary_cross` | สรุปเป้าหมายการเงิน | Expense |
| `get_income_summary_cross` | สรุปรายได้ | Planning |
| `get_tax_summary_cross` | สรุปการยื่นภาษี | Investment |

### ปุ่มทดสอบ

#### 1.1 "วิเคราะห์ภาษีพร้อมดูพอร์ตการลงทุน"

**Intent ที่คาดหวัง:** `tax`

**Flow:**
```
User → Router (classify → "tax") → Tax Agent
         Tax Agent เรียก tools:
         ├── calculate_thai_tax (คำนวณภาษี)
         └── get_portfolio_summary_cross (ดึงข้อมูลพอร์ตจาก Investment domain)
```

**สิ่งที่ทดสอบ:**
- Router จำแนก intent เป็น "tax" ได้ถูกต้อง
- Tax Agent เรียก cross-agent tool `get_portfolio_summary_cross` เพื่อดึงข้อมูลพอร์ตการลงทุน
- Tax Agent นำข้อมูลพอร์ตมาวิเคราะห์ร่วมกับข้อมูลภาษี (เช่น กำไรจากหุ้นที่ต้องเสียภาษี)

**ตัวอย่างผลลัพธ์ที่คาดหวัง:**
- สรุปสถานะภาษีของผู้ใช้
- แสดงข้อมูลพอร์ตการลงทุน (มูลค่า, กำไร/ขาดทุน)
- วิเคราะห์ผลกระทบทางภาษีจากการลงทุน (ถ้ามี)

---

#### 1.2 "วางแผนการเงิน ดูรายจ่ายและรายได้"

**Intent ที่คาดหวัง:** `planning`

**Flow:**
```
User → Router (classify → "planning") → Planning Agent
         Planning Agent เรียก tools:
         ├── view_financial_goals (ดูเป้าหมายการเงิน)
         ├── get_expense_summary_cross (ดึงค่าใช้จ่ายจาก Expense domain)
         └── get_income_summary_cross (ดึงรายได้จาก Income domain)
```

**สิ่งที่ทดสอบ:**
- Router จำแนก intent เป็น "planning" ได้ถูกต้อง
- Planning Agent เรียก cross-agent tools 2 ตัว: `get_expense_summary_cross` + `get_income_summary_cross`
- Planning Agent นำข้อมูลค่าใช้จ่ายและรายได้มาวิเคราะห์ร่วมกับเป้าหมายการเงิน

**ตัวอย่างผลลัพธ์ที่คาดหวัง:**
- แสดงเป้าหมายการเงินของผู้ใช้
- สรุปรายจ่ายรายเดือน + รายได้รายปี
- วิเคราะห์ว่ารายจ่ายและรายได้สอดคล้องกับเป้าหมายหรือไม่

---

#### 1.3 "ดูค่าใช้จ่ายเทียบเป้าหมาย"

**Intent ที่คาดหวัง:** `expense`

**Flow:**
```
User → Router (classify → "expense") → Expense Agent
         Expense Agent เรียก tools:
         ├── get_monthly_summary (สรุปรายจ่ายเดือนนี้)
         └── get_goals_summary_cross (ดึงเป้าหมายจาก Planning domain)
```

**สิ่งที่ทดสอบ:**
- Router จำแนก intent เป็น "expense" ได้ถูกต้อง
- Expense Agent เรียก cross-agent tool `get_goals_summary_cross` เพื่อดึงเป้าหมายการเงิน
- Expense Agent เปรียบเทียบค่าใช้จ่ายกับเป้าหมายที่ตั้งไว้

**ตัวอย่างผลลัพธ์ที่คาดหวัง:**
- สรุปค่าใช้จ่ายรายเดือน (แยกตามหมวด)
- แสดงเป้าหมายที่เกี่ยวข้อง (เช่น เป้าออมเงิน)
- วิเคราะห์ว่าค่าใช้จ่ายกระทบเป้าหมายอย่างไร

---

## 2. ทดสอบคำแนะนำเชิงรุก (Proactive Recommendations)

### แนวคิด

Recommendation Agent ใช้สถาปัตยกรรม **Rule-Based + LLM Hybrid**:

1. **Service Layer (Rule-Based):** ดึงข้อมูลทุก 5 โดเมน → วิเคราะห์ตามกฎ 10 ข้อ → สร้าง `Recommendation` objects
2. **Agent Layer (LLM):** รับ recommendations → สรุปเป็นภาษาไทยที่เข้าใจง่าย

### กฎวิเคราะห์ 10 ข้อ

| กฎ | หมวด | Priority | เงื่อนไขที่ trigger |
|---|---|---|---|
| ค่าใช้จ่ายกระจุกตัว | ลดค่าใช้จ่าย | 3 (สูง) | หมวดใดหมวดหนึ่ง > 40% ของรายจ่ายทั้งหมด |
| ใช้จ่ายเกินตัว | อัตราการออม | 4 (สูงมาก) | ค่าใช้จ่าย > 80% ของรายได้ต่อเดือน |
| อัตราออมต่ำ | อัตราการออม | 3 (สูง) | ออม < 20% ของรายได้ |
| ลดหย่อนภาษีน้อย | ลดหย่อนภาษี | 4 (สูงมาก) | ใช้สิทธิ์ลดหย่อน < 50% ของที่ใช้ได้ |
| ยังไม่ได้ยื่นภาษี | ลดหย่อนภาษี | 5 (เร่งด่วน) | ไม่พบข้อมูลการยื่นภาษี |
| พอร์ตกระจุกตัว | ปรับพอร์ต | 3 (สูง) | หุ้นตัวเดียว > 30% ของพอร์ต |
| พอร์ตขาดทุน | ปรับพอร์ต | 2 (ปานกลาง) | กำไร/ขาดทุนรวม < 0 |
| เป้าหมายล่าช้า | ติดตามเป้าหมาย | 4 (สูงมาก) | ความคืบหน้า < 50% |
| ไม่มีเงินสำรองฉุกเฉิน | เงินสำรองฉุกเฉิน | 5 (เร่งด่วน) | ไม่มีเป้าหมายประเภท emergency_fund |
| คะแนนสุขภาพการเงิน | — | — | คำนวณจาก 100 − (priority × 5 ต่อคำแนะนำ) |

### ปุ่มทดสอบ

#### 2.1 "วิเคราะห์การเงินและแนะนำทั้งหมด"

**Intent ที่คาดหวัง:** `recommendation`

**Flow:**
```
User → Router (classify → "recommendation") → Recommendation Agent
         Recommendation Agent เรียก tools:
         └── generate_financial_recommendations
               ├── gather_all_financial_data
               │     ├── get_expense_summary (Expense domain)
               │     ├── get_portfolio_summary (Investment domain)
               │     ├── get_goals_summary (Planning domain)
               │     ├── get_income_summary (Income domain)
               │     └── get_tax_filing_summary (Tax domain)
               ├── analyze_expense_patterns → ตรวจกฎ 1-2
               ├── analyze_tax_optimization → ตรวจกฎ 4-5
               ├── analyze_investment_risk → ตรวจกฎ 6-7
               ├── analyze_goal_progress → ตรวจกฎ 8
               ├── analyze_savings_rate → ตรวจกฎ 3, 9
               └── calculate_health_score → คำนวณคะแนน
         Recommendation Agent สรุปผลเป็นภาษาไทย
```

**สิ่งที่ทดสอบ:**
- Router จำแนก intent เป็น "recommendation" ได้ถูกต้อง
- Recommendation Agent เรียก `generate_financial_recommendations` tool
- Service layer ดึงข้อมูลจากทุก 5 โดเมน (expense, portfolio, goals, income, tax)
- กฎทั้ง 10 ข้อถูกประเมินจากข้อมูลจริงของผู้ใช้
- คำแนะนำถูกเรียงตาม priority (สูงสุดก่อน)
- LLM สรุปคำแนะนำเป็นภาษาไทยที่เข้าใจง่าย

**ตัวอย่างผลลัพธ์ที่คาดหวัง:**
- จำนวนคำแนะนำทั้งหมด (เช่น "พบ 3 คำแนะนำ")
- คะแนนสุขภาพการเงิน (0-100)
- คำแนะนำแต่ละข้อพร้อมหมวด, priority, คำอธิบาย, สิ่งที่ควรทำ
- (สำหรับผู้ใช้ใหม่ที่ไม่มีข้อมูล จะได้คำแนะนำพื้นฐาน เช่น "ไม่มีเงินสำรองฉุกเฉิน" และ "ยังไม่ได้ยื่นภาษี")

---

#### 2.2 "ตรวจสุขภาพการเงินของฉัน"

**Intent ที่คาดหวัง:** `recommendation`

**Flow:**
```
User → Router (classify → "recommendation") → Recommendation Agent
         Recommendation Agent เรียก tools:
         └── get_financial_health_score
               ├── generate_recommendations (เรียก service layer เดียวกัน)
               └── return {health_score, total_recommendations, top_issues}
```

**สิ่งที่ทดสอบ:**
- Router จำแนก intent เป็น "recommendation" ได้ถูกต้อง
- Recommendation Agent เรียก `get_financial_health_score` tool (แทน `generate_financial_recommendations`)
- ผลลัพธ์มีเฉพาะคะแนนสรุปและ top 3 ปัญหา (ไม่ใช่รายงานเต็ม)
- LLM อธิบายคะแนนและปัญหาหลักเป็นภาษาไทย

**ตัวอย่างผลลัพธ์ที่คาดหวัง:**
- คะแนนสุขภาพการเงิน (เช่น "คะแนน: 50/100")
- จำนวนคำแนะนำที่พบ
- Top 3 ปัญหาเร่งด่วน (เช่น "ยังไม่ได้ยื่นภาษี", "ไม่มีเงินสำรองฉุกเฉิน")

---

## สถาปัตยกรรมโดยรวม

```
                    ┌──────────────┐
      User Input →  │ Router Agent │  → classify intent
                    └──────┬───────┘
                           │
         ┌─────────┬───────┼────────┬─────────────┐
         ▼         ▼       ▼        ▼             ▼
      Tax Agent  Expense  Invest  Planning  Recommendation
         │       Agent    Agent    Agent       Agent
         │         │       │        │             │
         │         │       │        │    ┌────────┴────────┐
         │         │       │        │    │ Service Layer    │
         │         │       │        │    │ (Rule-Based)     │
         │         │       │        │    │ ┌──────────────┐ │
         │         │       │        │    │ │ gather_all_  │ │
         │         │       │        │    │ │ financial_   │ │
         │         │       │        │    │ │ data()       │ │
         │         │       │        │    │ └──────┬───────┘ │
         │         │       │        │    │        │         │
         │         │       │        │    │   5 domains      │
         │         │       │        │    │   10 rules       │
         │         │       │        │    │   health score   │
         │         │       │        │    └─────────────────┘ │
         │         │       │        │                        │
         └────┬────┴───┬───┴────┬───┴────────────────────────┘
              │        │        │
              ▼        ▼        ▼
         Cross-Agent Tools (ดึงข้อมูลข้ามโดเมน)
              │        │        │
              ▼        ▼        ▼
         ┌─────────────────────────┐
         │     Database (SQLite)   │
         │  User, Expense, Income, │
         │  Tax, Investment, Goals │
         └─────────────────────────┘
```

## ความแตกต่างระหว่าง Multi-Agent กับ Recommendation

| | Multi-Agent | Recommendation |
|---|---|---|
| **จุดประสงค์** | ตอบคำถามเฉพาะทาง + ดึงข้อมูลข้ามโดเมน | วิเคราะห์ภาพรวมทั้งหมด + ให้คำแนะนำ |
| **Agent ที่ทำงาน** | Agent เดียว (Tax/Expense/Investment/Planning) | Recommendation Agent |
| **ข้อมูลที่ใช้** | โดเมนหลัก + cross-agent tool (1-2 โดเมนเสริม) | ทุก 5 โดเมนพร้อมกัน |
| **ผลลัพธ์** | คำตอบเฉพาะเรื่อง | รายงานคำแนะนำ + คะแนนสุขภาพการเงิน |
| **กฎวิเคราะห์** | ไม่มี (LLM วิเคราะห์เอง) | 10 กฎ rule-based + LLM สรุป |
