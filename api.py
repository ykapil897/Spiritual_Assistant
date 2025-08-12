import psutil, os

# def print_memory_usage(stage=""):
#     process = psutil.Process(os.getpid())
#     mem = process.memory_info().rss / (1024 * 1024)  # in MB
#     print(f"[MEMORY {stage}] Used: {mem:.2f} MB")

# print_memory_usage("After import psutil and os")
# print_memory_usage("Start memory usage")
from fastapi import FastAPI, Depends, HTTPException
# print_memory_usage("After import fastapi")
from pydantic import BaseModel
# print_memory_usage("After import pydantic")
from sqlalchemy.orm import Session
# print_memory_usage("After import sqlalchemy")
from db_manager import get_db, init_db, get_user_by_username, store_query, store_quiz_result, store_verse_read, create_or_update_user
# print_memory_usage("After import db_manager")
from rag_testing_final import PrabhupadaRAG
# print_memory_usage("After import rag_testing_final")
from quiz_generator import QuizGenerator
# print_memory_usage("After import quiz_generator")
# print_memory_usage("After import psutil and os")


app = FastAPI()
init_db()

# print_memory_usage("After init_db")
@app.get("/")
def hello():
    return {"message": "Hello, world!"}

class QueryRequest(BaseModel):
    username: str
    query: str


class QuizRequest(BaseModel):
    username: str
    topic: str
    quiz_type: str
    difficulty: str
    num_questions: int


class QuizSubmission(BaseModel):
    username: str
    quiz_data: dict
    user_answers: dict


class PreferenceUpdate(BaseModel):
    username: str
    prabhupada_ratio: int
    answer_length: str
    answer_format: str
    devotee_level: str
    quiz_type: str
    difficulty: str
    num_questions: int

# print_memory_usage("After defining models")

@app.post("/process_query")
def process_query(data: QueryRequest, db: Session = Depends(get_db)):
    user = get_user_by_username(db, data.username)
    if not user:
        raise HTTPException(status_code=404, detail="User not found")

    preferences = {
        "prabhupada_ratio": int(user.prabhupada_ratio),
        "answer_length": user.answer_length,
        "answer_format": user.answer_format,
        "devotee_level": user.devotee_level
    }

    # print_memory_usage("Before model load")
    rag_model = PrabhupadaRAG()
    # print_memory_usage("After rag_model")

    final_answer = rag_model.process_query(data.query, preferences["prabhupada_ratio"],
                                           preferences["answer_length"], preferences["answer_format"],
                                           preferences["devotee_level"])
    store_query(db, user.id, data.query, final_answer)
    return {"answer": final_answer}


@app.post("/generate_quiz")
def generate_quiz(data: QuizRequest, db: Session = Depends(get_db)):
    user = get_user_by_username(db, data.username)
    if not user:
        raise HTTPException(status_code=404, detail="User not found")

    # print_memory_usage("Before model load")
    quiz_gen = QuizGenerator()
    # print_memory_usage("After quiz_gen")

    quiz = quiz_gen.generate_quiz(
        query=data.topic,
        quiz_type=data.quiz_type,
        difficulty=data.difficulty,
        num_questions=data.num_questions
    )
    return quiz


@app.post("/submit_answers")
def submit_answers(submission: QuizSubmission, db: Session = Depends(get_db)):
    user = get_user_by_username(db, submission.username)
    if not user:
        raise HTTPException(status_code=404, detail="User not found")

    # print_memory_usage("Before model load")
    quiz_gen = QuizGenerator()
    # print_memory_usage("After quiz_gen")

    results = quiz_gen.check_answers(submission.quiz_data, submission.user_answers)

    questions = [q.get("statement") or q.get("question") for q in submission.quiz_data["questions"]]
    correct = [
        q["options"][q["answer"]] if submission.quiz_data["meta"]["quiz_type"] == "mcq" else q["answer"]
        for q in submission.quiz_data["questions"]
    ]
    user_ans = [submission.user_answers.get(str(i + 1)) for i in range(len(questions))]
    references = [q.get("reference") for q in submission.quiz_data["questions"]]

    store_quiz_result(
        db, user.id,
        quiz_type=submission.quiz_data["meta"]["quiz_type"],
        difficulty=submission.quiz_data["meta"]["difficulty"],
        questions=questions,
        correct_answers=correct,
        user_answers=user_ans,
        total_correct=results["correct_count"],
        references=references
    )

    return results


@app.post("/update_preferences")
def update_preferences(pref: PreferenceUpdate, db: Session = Depends(get_db)):
    updated_user = create_or_update_user(db, pref.dict())
    return {"status": "Preferences updated", "username": updated_user.username}


@app.get("/get_history/{username}")
def get_history(username: str, db: Session = Depends(get_db)):
    user = get_user_by_username(db, username)
    if not user:
        raise HTTPException(status_code=404, detail="User not found")

    queries = [{"query": q.query, "answer": q.answer, "timestamp": q.timestamp} for q in user.queries]
    quizzes = [{
        "quiz_type": q.quiz_type,
        "difficulty": q.difficulty,
        "total_correct": q.total_correct,
        "questions": q.question,
        "user_answers": q.user_answer,
        "correct_answers": q.correct_answer,
        "references": q.references,
        "timestamp": q.timestamp
    } for q in user.quizzes]

    verses = [{"verse": v.verse, "source": v.source, "timestamp": v.timestamp} for v in user.verses]

    return {
        "queries": queries,
        "quizzes": quizzes,
        "verses": verses
    }

