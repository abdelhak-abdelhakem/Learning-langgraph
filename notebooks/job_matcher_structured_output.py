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

job_description = """# Job 2 : About the job - DRIVECO

## Qui sommes-nous ? 👀
En tant qu’acteur innovant de premier plan dans le domaine de la recharge de véhicules électriques, la mission de DRIVECO est de rendre la mobilité électrique accessible à tous. Nous nous efforçons d’offrir la meilleure expérience utilisateur, de bâtir le réseau le plus fiable et de recourir aux énergies renouvelables pour construire une société plus verte et plus durable.

DRIVECO, c’est un réseau de 10 000 points de charge en service ou en cours de déploiement, et le plus grand réseau de bornes de recharge ouvertes au public en France, reconnu pour sa fiabilité exceptionnelle. Depuis sa création, DRIVECO a permis de recharger l’équivalent de 230 millions de kilomètres parcourus en véhicules électriques, soit 5 500 fois le tour de la Terre. Avec plus de 45 000 tonnes de CO₂ évitées, DRIVECO a fourni plus de 40 millions de kWh d’électricité depuis sa fondation.

👉 Pour en savoir plus, rendez-vous sur www.driveco.com ou sur notre page LinkedIn (DRIVECO).

## À propos du poste 🚀
* **Localisation:** Paris (HQ) - télétravail 1 j / semaine.
* **Rémunération :**
    * Master 1 : 1000 euros brut par mois.
    * Master 2 : 1250 euros brut par mois.
* **Langue du poste :** Français et Anglais.

## Présentation Du Poste
Driveco recherche un(e) stagiaire IA pour concevoir et déployer des workflows, agents et automatisations internes. Rattaché(e) au CTPO, tu travailleras avec les équipes Tech, Product, Data, Ops, Sales et Marketing pour identifier les besoins, prioriser les cas d’usage, prototyper des solutions et mesurer leur impact.

Les sujets pourront couvrir des assistants métiers, bases de connaissance, automatisation documentaire, reporting, support IA, génération de contenus, extraction de données ou connexion d’outils internes.

Nous cherchons une personne curieuse, structurée et pragmatique, avec de bonnes bases techniques, un fort intérêt pour l’IA générative et l’envie de transformer des irritants métiers en solutions concrètes. Il n’est pas nécessaire d’être expert(e) : le stage est fait pour apprendre, expérimenter et livrer des cas d’usage utiles.

## Contexte
DRIVECO souhaite accélérer l’usage de l’IA dans ses équipes pour gagner en efficacité, automatiser les tâches répétitives et mieux exploiter la connaissance interne.

L’objectif du stage n’est pas de produire des démonstrations isolées, mais de transformer des besoins métiers réels en workflows IA simples, utiles et adoptés par les équipes.

## Mission
Rattaché(e) au CTPO et au VP Product & Data, tu travailleras avec les équipes Tech, Product, Data, Ops, Sales, Marketing et fonctions support pour identifier, cadrer, prototyper et documenter des cas d’usage IA.

La mission principale : partir d’un irritant métier, le formaliser, construire une première solution IA, la tester avec les utilisateurs, puis mesurer si elle apporte un vrai gain.

## Responsabilités
1. **Identifier et cadrer les cas d’usage :**
    * Échanger avec les équipes pour comprendre leurs tâches récurrentes, leurs pertes de temps et leurs besoins d’automatisation.
    * Aider à prioriser les cas d’usage selon l’impact attendu, la simplicité de mise en œuvre et l’adoption probable.
    * Rédiger des fiches use case claires : problème, utilisateurs, données nécessaires, solution cible, risques, indicateurs de succès.
2. **Prototyper des workflows IA :**
    * Construire des premiers workflows à l’aide d’outils IA, d’APIs, de scripts ou d’outils no-code/low-code.
    * Créer des assistants ou agents capables de rechercher, résumer, extraire, structurer, générer ou préparer des actions.
    * Tester les résultats, améliorer les prompts, clarifier les règles métier et simplifier l’expérience utilisateur.
3. **Connecter les outils et automatiser les actions :**
    * Explorer les intégrations possibles avec les outils internes pertinents.
    * Automatiser certaines étapes de workflow : génération de documents, synthèse de demandes, mise en forme de données, notifications, reporting.
    * Travailler avec les équipes Tech, Product ou Data lorsque le sujet nécessite une intégration plus robuste.
4. **Accompagner le déploiement :**
    * Documenter les solutions mises en place de façon simple et réutilisable.
    * Aider les équipes à prendre en main les workflows.
    * Mesurer l’usage, collecter les retours et proposer les améliorations nécessaires.

## Profil recherché
Nous cherchons une personne curieuse, structurée et capable d’apprendre vite. Il n’est pas attendu de tout maîtriser avant le stage : l’important est de savoir creuser un sujet, tester rapidement et transformer un besoin métier en solution concrète.

### Compétences Utiles
* Connaissance en Python ou JavaScript/TypeScript.
* Compréhension du fonctionnement des APIs et des outils SaaS.
* Intérêt marqué pour les LLMs, les agents IA, le prompt engineering et l’automatisation.
* Capacité à structurer un problème, rédiger clairement et documenter son travail.
* Envie de discuter avec des utilisateurs métiers pour comprendre leurs vrais besoins.

### Ce qui serait un plus
* Avoir déjà testé des APIs IA ou construit un petit assistant personnel.
* Connaître des outils comme n8n, Zapier, OpenAI API, Claude, Hermes ou OpenClaw.
* Avoir des notions de RAG, embeddings, recherche sémantique, data visualisation ou BI.
* Avoir déjà automatisé une tâche personnelle, académique ou associative.

### Qualités attendues
* Autonomie et capacité à avancer avec un cadre clair mais incomplet.
* Pragmatisme : privilégier une solution simple qui marche plutôt qu’un prototype trop complexe.
* Rigueur sur la qualité des données, la confidentialité et la documentation.
* Bonne communication avec des profils techniques et non techniques.
* Esprit produit : comprendre le problème avant de choisir l’outil.

## Pourquoi vous allez adorer nous rejoindre 🥰
Rejoignez DRIVECO et bénéficiez de notre programme de qualité de vie au travail. Télétravail, congés généreux, avantages du CSE… tout est mis en œuvre pour favoriser votre épanouissement professionnel. Intégrez une entreprise engagée sur les plans social et environnemental !

## Votre Processus De Recrutement
1. 👓 Tests en ligne sur la plateforme TestGorilla.
2. 💡 Un entretien de découverte avec Sandy notre Talent Acquisition Manager (30 en visio).
3. 📍 Un entretien avec Sofiane, CTPO (1h dans nos bureaux idéalement).
4. 🌟 Vérifications RH et proposition d’embauche !

## Nos engagements
DRIVECO s’engage en faveur de l’égalité des chances et valorise la diversité en assurant un environnement inclusif pour tous. Nous encourageons les candidatures de toutes les minorités, identités de genre, orientations sexuelles, et sommes attentifs aux besoins spécifiques des personnes en situation de handicap ou nécessitant des aménagements particuliers.

Même si vous ne cochez pas toutes les cases, n’hésitez pas à postuler : discutons-en ensemble. 😊"""

my_profile = """# Abdelhak ABDELHAKEM
**Étudiant Ingénieur IA | Spécialiste LLM & RAG Systems**

- **Email :** abdelhakemabdelhak@gmail.com
- **Téléphone :** +213 790 605 350
- **Localisation :** Sidi Bel Abbès, Algérie
- **GitHub :** github.com/abdelhak-abdelhakem
- **LinkedIn :** linkedin.com/in/abdelhak-abdelhakem

## Profil
Étudiant en 4e année Ingénierie IA (Bac+5, équivalent Master), spécialisé en LLM Engineering et Retrieval-Augmented Generation.
Deux systèmes RAG complets (un projet personnel et un déploiement en environnement entreprise) évalués scientifiquement via RAGAS.
Recherche un stage PFE de 6 mois (février 2027) en France dans un environnement LLM appliqué.

## Expérience

### Stagiaire Ingénieur IA | Développement RAG Enterprise
**Groupe des Sociétés Hasnaoui, Sidi Bel Abbès** | *Fév. – Juin 2026*
Stage encadré de 15 jours, développement poursuivi en autonomie sur le semestre. Architecture adaptée depuis UniBot DZ (projet personnel). Conception et développement individuel à 100% du système technique.
- **Pipeline hybride complet :** Query Rewriting → BM25 + ChromaDB → Cohere rerank-v3.5 → LLM Streaming (SSE)
- **Évaluation RAGAS :** Context Precision 0.892 | Context Recall 0.812 | Faithfulness 0.769 | Answer Relevancy 0.702 (50 paires Q&A : arabe, français, anglais)
- **Observabilité LangSmith end-to-end :** $0.00079/requête | 4011 tokens/requête | latence 17-20s mesurée par étape
- **Frontend & Backend :** React + Tailwind avec streaming SSE | Backend FastAPI (8 endpoints, upload/suppression de documents en temps réel)
- **Déploiement :** Ubuntu Server, README professionnel, diagramme d'architecture, documentation complète.

## Recherche

### Membre de Recherche | Architectures SSM & Mamba
**AI Quantum Community (AiQC) | Club de recherche à l'ESI** | *Mai 2026 – Présent*
- Étude comparative des State Space Models (SSMs) et l'architecture Mamba face aux Transformers (complexité linéaire).
- Analyse des mécanismes de Selective Scan pour optimiser le traitement de contextes ultra-longs dans les systèmes de génération.
- Exploration de l'efficacité computationnelle (hardware-aware) pour réduire l'empreinte mémoire lors de l'inférence.

## Projets

### UniBot DZ | Chatbot RAG Personnel
*github.com/abdelhak-abdelhakem/my_rag_chatbot*
- Projet personnel initié de manière autonome, développé sur 8 documents PDF (premier système RAG complet).
- RAGAS Answer Relevancy 0.921 (92e percentile), +7.4% d'amélioration via tests A/B systématiques (chunk sizes, k-values, poids hybrides).
- Architecture tri-interface : REST API FastAPI (Swagger, Pydantic) + UI Streamlit + CLI, conteneurisé Docker Compose.
- **Stack :** LangChain, FAISS, BM25, OpenAI API, RAGAS, PyMuPDF.
- Devenu la base technique de Hasnaoui Bot, adapté durant le stage chez Groupe des Sociétés Hasnaoui.

### Hasnaoui Bot | Chatbot RAG Enterprise Production
*github.com/abdelhak-abdelhakem*
- **Stack :** LangChain, FastAPI, React, Tailwind, ChromaDB, BM25, Cohere, OpenAI API, RAGAS, LangSmith, PyMuPDF.
- **Scores RAGAS :** Context Precision 0.892 | Context Recall 0.812 | Faithfulness 0.769 | Answer Relevancy 0.702.
- Multilingue natif (AR/FR/EN) : détection automatique de langue, Cohere choisi pour support arabe natif.

### Pipeline NLP Multi-Tâches
- Analyse de sentiment, traduction, QA extractive, résumé automatique sur corpus d'avis automobiles.
- **Stack :** Hugging Face Transformers, évaluation ROUGE, BERT Fine-Tuning, pipeline unifié Python.

## Compétences Techniques
- **LLM & RAG :** LangChain, Mamba, SSM, OpenAI API, Cohere, RAG, Prompt Engineering, RAGAS.
- **Embeddings :** ChromaDB, FAISS, BM25, Hybrid Search, Cohere Reranking, Sentence Transformers.
- **ML/NLP :** Hugging Face Transformers, PyTorch, Fine-Tuning, LoRA/QLoRA, BERT, ROUGE, BERTScore.
- **Backend :** Python 3.10+, FastAPI, Flask, React, Streamlit, SSE Streaming, REST APIs, Pydantic.
- **MLOps :** Docker, Docker Compose, Git/GitHub, LangSmith, Weights & Biases, Linux/Bash.
- **Data / Dist. :** Pandas, NumPy, PySpark, Ray, Dask, SQL.
- **Cloud / Infra :** AWS (notions), Ubuntu Server, WSL2.

## Formation
**Diplôme Ingénieur en Intelligence Artificielle (Bac+5, équivalent Master)** | *2022 – 2027 (prévu)*
*Université Djilali Liabès, Sidi Bel Abbès*
- Spécialisation : LLM Engineering, Systèmes RAG, Deep Learning, NLP.

## Distinctions
- **2e Place :** AIQC (ESI) Mini-Hackathon AI (Avril 2026)

## Certifications
- **Retrieval-Augmented Generation (RAG)** - DeepLearning.AI / Coursera (2025)
- **Hugging Face Transformers Course (NLP)** - Hugging Face (2025)

## Langues
- **Arabe :** Natif
- **Français :** B1 (objectif B2 oct. 2026)
- **Anglais :** B2"""


messages = [
    SystemMessage(content=f"You are a technical recruiter. Evaluate this candidate profile against the job description and return a structured assessment.\n\nCandidate profile:\n{my_profile}"),
    HumanMessage(content=f"Job description:\n{job_description}")
]


result = structured_llm.invoke(messages)
print(result)
print(type(result))  # should print <class '__main__.JobMatch'>