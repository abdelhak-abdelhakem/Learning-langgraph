from pydantic import BaseModel
from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage, SystemMessage
from dotenv import load_dotenv 
import os


load_dotenv()
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

llm = ChatOpenAI(
    model="gpt-5-nano", temperature = 0) 


class JobMatch (BaseModel):
    match_score: int
    rationale: str
    matched_skills: list[str]
    missing_skills: list[str]

structured_llm = llm.with_structured_output(JobMatch)

job_description = """About the job

Lysi est un cabinet de conseil en stratégie durable qui accompagne les entreprises, les institutions financières et les organisations publiques à intégrer les enjeux de la transition socio-écologique au cœur de leur stratégie.

Fondé en 2024 par Constantin de Salvatore, Thomas Nouvian et Sylvain Prévot, Lysi est basé à Marseille et a vocation à travailler partout en France et en Europe.

 

Notre raison d’agir

Lysi est née de la conviction forte qu’il y a urgence à agir face aux dérèglements environnementaux auxquels nous faisons face : multiplication des phénomènes climatiques extrêmes, effondrement critique de la biodiversité, explosion des déchets plastiques, etc. A côté de ces enjeux écologiques, impossible d’aborder la transition sans son volet social et sociétal. Elle doit être désirable, accessible à tous et ne laisser personne de côté.


Notre mission

Entreprise à mission, Lysi a pour raison d’être d’intensifier la mise en action des acteurs économiques pour dynamiser la transition socio-écologique, nécessaire à l’avènement d’une société prospère, dans le respect de ce que la planète peut nous offrir.

 

L’IA Générative au service de la transition

Nous considérons l’IA générative comme une opportunité pour accélérer les projets de durabilité. Chez Lysi, nous utilisons au quotidien des outils d'IA générative permettant de délivrer des missions de qualité, plus rapidement et à moindre coût. Conscients des enjeux autour de cette technologie, nous mettons un point d’honneur à faire une utilisation responsable, éthique et sécurisée.


Pourquoi rejoindre Lysi ?

Honnêteté intellectuelle, rigueur, plaisir à faire du travail de très haute facture, vous rejoignez une équipe engagée, soudée, passionnée et ambitieuse qui a à cœur de faire grandir, progresser et responsabiliser les étudiants.

C'est aussi l'opportunité de lier IA générative et durabilité, un des sujets les plus stratégiques des décennies à venir.


Votre mission

Structurer les pratiques IA de Lysi : formaliser nos usages, définir les standards et poser les fondations d'une culture IA partagée par toute l'équipe.

Harmoniser les process et workflows internes : cartographier, optimiser et automatiser les flux de production récurrents sur l'ensemble de nos offres de conseil.

Déployer des outils IA en production : des agents et pipelines réels, utilisés au quotidien par les consultants et dans les missions clients, pas des prototypes.

Co-construire des offres dédiées IA × Durabilité : contribuer à la définition et à la formalisation de nouvelles offres de conseil intégrant l'IA générative comme levier différenciant.


Profil

Bac +5 ou césure : dernière année de grande école d’ingénieur, grande école de commerce (PGE), Sciences Po

Vous êtes doté d’une très grande rigueur, d’une forte capacité d’analyse et d’un esprit de synthèse

Vous avez une forte capacité à résoudre des problèmes complexes

Vous êtes curieux, proactif et autonome

Vous démontrez un intérêt fort pour les thématiques du développement durable et de l'IA

Vous aimez le travail en équipe, disposez d’un excellent relationnel et aimez contribuer à une ambiance de travail enthousiasmante


Hard Skills

Expérience pratique avec les LLMs : prompt engineering, function calling, gestion du contexte et des tokens.

Compréhension des architectures RAG

Notions d'orchestration agentique 


Soft skills & mindset

Profil "builder" : vous êtes là pour tester et déployer

Intérêt authentique et documenté pour les enjeux ESG, climatiques et de durabilité.

Capacité à faire le pont entre une logique métier (consultant RSE) et une contrainte technique (architecture IA).

Autonomie, rigueur et appétence pour les environnements de travail exigeants

Goût pour la documentation et la transmission : vos workflows doivent être utilisables par des non-développeurs.


Atouts différenciants

Connaissance des référentiels ESG : ESRS, GHG Protocol, VSME, GRI, SBTi, taxonomie EU.

Expérience sur des outils no-code / low-code d'automatisation (Make, n8n, Zapier).

Projets personnels ou académiques impliquant des agents IA ou des pipelines RAG (GitHub bienvenu).

Double culture ingénieur + sciences humaines / management.


Alors ce stage est fait pour vous !


Conditions

Type de contrat : stage de fin d’études ou césure

Durée : 6 mois à compter de septembre 2026

Rémunération attractive selon profil 

+ 50 % pass RTM 

Titres restaurant

Semaine de 4,5 jours, le vendredi aprem c'est off !

Congés : 1 jour / mois

Lieu du stage : 21 rue Haxo 13001 Marseille, un cadre de travail particulièrement attractif dans un écosystème stimulant avec :

Salle de sport

Local vélo

Rooftop !

 

Nous avons à cœur de favoriser la diversité et l’inclusion dans nos recrutements, nos locaux sont parfaitement équipés pour accueillir les personnes à mobilité réduite.

 

Merci d’envoyer CV et lettre de motivation à contact@lysi.eco"""

my_profile = """Abdelhak ABDELHAKEM
Étudiant Ingénieur IA · Spécialiste LLM & RAG Systems
abdelhakemabdelhak@gmail.com · +213 790 605 350 · Sidi Bel Abbès, Algérie · github.com/abdelhak-abdelhakem ·
linkedin.com/in/abdelhak-abdelhakem
Proﬁl
Étudiant en 4e année Ingénierie IA (Bac+5, équivalent Master), spécialisé en LLM Engineering et Retrieval-
Augmented Generation. Deux systèmes RAG complets — un projet personnel et un déploiement en environnement
entreprise — évalués scientiﬁquement via RAGAS. Recherche un stage PFE de 6 mois (février 2027) en France
dans un environnement LLM appliqué.
Expérience
Stagiaire Ingénieur IA — Développement RAG Enterprise
Fév. – Juin 2026
Groupe des Sociétés Hasnaoui · Sidi Bel Abbès
Stage encadré de 15 jours, développement poursuivi en autonomie sur le semestre · Architecture adaptée depuis
UniBot DZ (projet personnel) · Conception et développement à 100 % du système technique au sein d’une équipe
de 5
Pipeline hybride complet : Query Rewriting → BM25 + ChromaDB → Cohere rerank-v3.5 → LLM Streaming
(SSE)
• Évaluation RAGAS : Context Precision 0.892 · Context Recall 0.812 · Faithfulness 0.769 · Answer Relevancy
0.702 — 50 paires Q&A (arabe, français, anglais)
• Observabilité LangSmith end-to-end : $0.00079/requête · 4 011 tokens/requête · latence 17–20 s mesurée par
étape
• Frontend React + Tailwind avec streaming SSE · Backend FastAPI — 8 endpoints · upload/suppression de
documents en temps réel
• Déploiement Ubuntu Server · README professionnel · diagramme d’architecture · documentation complète
•
Recherche
Membre de Recherche — Architectures SSM & Mamba
Mai 2026 – Présent
AI Quantum Community (AiQC) · Club de recherche à l’École Supérieure en Informatique (ESI)
Étude comparative des State Space Models (SSMs) et de l’architecture Mamba face aux Transformers (com-
plexité linéaire)
• Analyse des mécanismes de Selective Scan pour optimiser le traitement de contextes ultra-longs dans les systèmes
de génération
• Exploration de l’eﬃcacité computationnelle (hardware-aware) pour réduire l’empreinte mémoire lors de l’inférence
•
Projets
UniBot DZ — Chatbot RAG Personnel
github.com/abdelhak-abdelhakem/my_rag_chatbot
Projet personnel initié de manière autonome, développé sur 8 documents PDF — premier système RAG complet
RAGAS Answer Relevancy 0.921 (92e percentile) · +7.4 % d’amélioration via tests A/B systématiques (chunk
sizes, k-values, poids hybrides)
• Architecture tri-interface : REST API FastAPI (Swagger, Pydantic) + UI Streamlit + CLI, conteneurisé Docker
Compose
•
••
•
Stack : LangChain · FAISS · BM25 · OpenAI API · RAGAS · PyMuPDF
Devenu la base technique de Hasnaoui Bot, adapté durant le stage chez Groupe des Sociétés Hasnaoui
Hasnaoui Bot — Chatbot RAG Enterprise Production
github.com/abdelhak-abdelhakem
Stack : LangChain · FastAPI · React · Tailwind · ChromaDB · BM25 · Cohere · OpenAI API · RAGAS ·
LangSmith · PyMuPDF
• Scores RAGAS : Context Precision 0.892 · Context Recall 0.812 · Faithfulness 0.769 · Answer Relevancy 0.702
• Multilingue natif (AR/FR/EN) — détection automatique de langue, Cohere choisi pour support arabe natif
•
Pipeline NLP Multi-Tâches
•
•
Analyse de sentiment · traduction · QA extractive · résumé automatique sur corpus d’avis automobiles
Stack : HuggingFace Transformers · évaluation ROUGE · BERT Fine-Tuning · pipeline uniﬁé Python
Compétences Techniques
LLM & RAG
Embeddings
ML / NLP
Backend
MLOps
Data / Dist.
Cloud / Infra
LangChain · Mamba · SSM · OpenAI API · Cohere · RAG · Prompt Engineering · RAGAS
ChromaDB · FAISS · BM25 · Hybrid Search · Cohere Reranking · Sentence Transformers
HuggingFace Transformers · PyTorch · Fine-Tuning LoRA/QLoRA · BERT · ROUGE ·
BERTScore
Python 3.10+ · FastAPI · Flask · React · Streamlit · SSE Streaming · REST APIs · Pydantic
Docker · Docker Compose · Git/GitHub · LangSmith · Weights & Biases · Linux/Bash
Pandas · NumPy · PySpark · Ray · Dask · SQL
AWS (notions) · Ubuntu Server · WSL2
Formation
Diplôme Ingénieur en Intelligence Artiﬁcielle — Bac+5 (équivalent Master)
2022 – 2027 (prévu)
Université Djilali Liabès · Sidi Bel Abbès · Spécialisation : LLM Engineering, Systèmes RAG, Deep Learning, NLP
Distinctions
2e Place — AiQC (ESI) Mini-Hackathon AI
Avril 2026
Certiﬁcations
Retrieval-Augmented Generation (RAG)
2025
DeepLearning.AI / Coursera
Hugging Face Transformers Course (NLP)
Hugging Face
Langues
Arabe (natif) · Français (B1, objectif B2 oct. 2026) · Anglais (B2)"""


messages = [
    SystemMessage(content=f"You are a technical recruiter. Evaluate this candidate profile against the job description and return a structured assessment.\n\nCandidate profile:\n{my_profile}"),
    HumanMessage(content=f"Job description:\n{job_description}")
]


result = structured_llm.invoke(messages)
print(result)
print(type(result))  # should print <class '__main__.JobMatch'>