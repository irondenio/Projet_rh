import streamlit as st
from PyPDF2 import PdfReader
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.messages import HumanMessage
import os

# Configuration de la page
st.set_page_config(page_title="IA Recrutement Pro 2026", layout="wide")

st.title("🤖 IA de Recrutement & Préparation d'Entretiens")
st.subheader("Auteur: Anthony DJOUMBISSI")
st.subheader("Analysez l'adéquation candidat-poste en un clic Version 1.0")
st.markdown("***Analysez les CVs par rapport aux fiches de poste et générez des questions d'entretien pertinentes grâce à gemini-2.5-flash.***")

# Sidebar pour la clé API
with st.sidebar:
    api_key = st.text_input("Clé API GenAI", type="password")
    st.info("Cet outil utilise gemini-2.5-flash pour l'analyse sémantique.")

# Zone de saisie
col1, col2 = st.columns(2)

with col1:
    job_description = st.text_area("Fiche de poste (Détails)", height=300, placeholder="Collez ici le descriptif du poste...")

with col2:
    uploaded_file = st.file_uploader("Charger le CV (Format PDF)", type="pdf")

# Traitement
if st.button("Lancer l'analyse IA"):
    if not api_key:
        st.error("Veuillez entrer votre clé API.")
    elif uploaded_file and job_description:
        with st.spinner("L'IA analyse le profil..."):
            # 1. Extraction du texte du PDF
            reader = PdfReader(uploaded_file)
            resume_text = ""
            for page in reader.pages:
                resume_text += page.extract_text()

            # 2. Préparation du Prompt pour l'Expert RH
            llm = ChatGoogleGenerativeAI(
                                         model="gemini-2.5-flash",
                                         google_api_key= api_key,  # "api_key" est la variable récupérée de st.text_input
                                         temperature=0
                                        )
            prompt = f"""
            En tant qu'expert en recrutement, analyse l'adéquation entre ce CV et cette fiche de poste.
            
            FICHE DE POSTE:
            {job_description}
            
            CV DU CANDIDAT:
            {resume_text}
            
            Fournis une réponse structurée :
            1. Score d'adéquation (sur 100).
            2. Points forts du candidat.
            3. Lacunes ou points à clarifier.
            4. Top 5 des questions d'entretien techniques à poser pour ce profil.
            """

            # 3. Appel à l'IA
            response = llm.invoke([HumanMessage(content=prompt)])
            
            # 4. Affichage des résultats
            st.success("Analyse terminée !")
            st.markdown("### 📊 Rapport d'Analyse RH")
            st.write(response.content)
            
            # Bouton de téléchargement du rapport
            st.download_button("Télécharger le rapport (TXT)", response.content, file_name="analyse_recrutement.txt")
    else:
        st.warning("Veuillez fournir à la fois le CV et la fiche de poste.")
