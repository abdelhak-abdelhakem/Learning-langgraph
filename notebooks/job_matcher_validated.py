from pydantic import BaseModel , Field , field_validator
from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage, SystemMessage
from dotenv import load_dotenv 
import os


load_dotenv()
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

llm = ChatOpenAI(
    model="gpt-5-nano", temperature = 0) 


class JobMatch (BaseModel):
    match_score: int = Field(...,
                             description="the macth score must be between 0 to 100")
    rationale: str 
    matched_skills: list[str] 
    missing_skills: list[str] 

    @field_validator("match_score")
    @classmethod
    def score_must_be_valid(cls, score:int)-> int:
        if not 0 <= score <= 100:
            raise ValueError(f"match_score must be between 0 and 100, got {score}")
        return score
    
    @field_validator("matched_skills", "missing_skills")
    @classmethod
    def no_empty_strings(cls, v):
        cleaned = [s for s in v if s.strip() != ""]
        return cleaned


structured_llm = llm.with_structured_output(JobMatch)

with open("notebooks/docs/my_profile.md", "r", encoding="utf-8") as file:
    my_profile = file.read()

job_postings = []
for i in range(1, 6):
    file_name = f"notebooks/docs/job{i}.md"
    
    try:
        with open(file_name, 'r', encoding='utf-8') as file:
            content = file.read()
            
            
            job_postings.append(content)
            print(f"Successfully added {file_name} to job_postings.")
            
    except FileNotFoundError:
        print(f"Warning: {file_name} was not found.")
        job_postings.append(None)





for i, job in enumerate(job_postings, 1):
    messages = [
    SystemMessage(content=f"You are a technical recruiter. Evaluate this candidate profile against the job description and return a structured assessment.\n\nCandidate profile:\n{my_profile}"),
    HumanMessage(content=f"Job description:\n{job}")
]


    result = structured_llm.invoke(messages)
    print(f"Job {i} — Score: {result.match_score}/100")
    print(f"Missing: {result.missing_skills}\n")