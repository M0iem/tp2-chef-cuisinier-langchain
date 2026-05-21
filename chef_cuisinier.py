"""
TP : Chef Cuisinier Personnel avec LangChain
Un agent intelligent qui propose des recettes selon les ingrédients disponibles,
avec mémoire des préférences et recherche web.
"""

import os
from dotenv import load_dotenv
from langchain.agents import create_agent
from langchain.messages import HumanMessage, SystemMessage
from langchain_ollama import ChatOllama
from langgraph.checkpoint.memory import InMemorySaver
from langchain.tools import tool
from tavily import TavilyClient

# Charger les variables d'environnement
load_dotenv()

# Initialiser Tavily pour la recherche web
tavily_client = TavilyClient(api_key=os.environ.get("TAVILY_API_KEY"))

# ==================== OUTILS (TOOLS) ====================

@tool("recherche_recette")
def rechercher_recette(requete: str) -> str:
    """
    Recherche des recettes ou techniques culinaires sur le web.
    Args:
        requete: La requête de recherche (ex: "recette de tarte aux pommes")
    """
    print(f"\n🔍 Recherche web : '{requete}'")
    try:
        resultats = tavily_client.search(requete, max_results=3)
        
        # Formater les résultats
        reponse = f"\n📚 Résultats de recherche pour '{requete}':\n\n"
        for i, resultat in enumerate(resultats.get('results', []), 1):
            reponse += f"{i}. {resultat.get('title', 'Sans titre')}\n"
            reponse += f"   {resultat.get('content', 'Pas de contenu')[:300]}...\n"
            reponse += f"   Source: {resultat.get('url', 'URL inconnue')}\n\n"
        
        return reponse
    except Exception as e:
        return f"Erreur de recherche : {e}"

# ==================== SYSTÈME PROMPT (Message de contrôle) ====================

system_prompt = """Tu es un chef cuisinier personnel professionnel, chaleureux et expert en cuisine.

RÈGLES IMPORTANTES :
1. Mémorise toutes les informations que l'utilisateur te donne (préférences alimentaires, allergies, ingrédients disponibles, niveau de cuisine)
2. Quand l'utilisateur donne des ingrédients, propose des recettes adaptées
3. Utilise l'outil 'recherche_recette' pour trouver de nouvelles recettes ou techniques si nécessaire
4. Demande des précisions sur les quantités ou ingrédients manquants si besoin
5. Propose toujours 1 à 3 plats selon les ingrédients disponibles
6. Donne les étapes clés de préparation pour le plat choisi
7. Sois enthousiaste et encourageant !

Ton rôle : Aider à cuisiner délicieusement avec ce qu'on a sous la main !"""

# ==================== INITIALISATION DU MODÈLE ====================

print("🍳 Initialisation du Chef Cuisinier Personnel...")
print("="*50)

# Modèle Ollama (local)
model = ChatOllama(
    model="llama3.2:3b",
    temperature=0.7,  # Un peu de créativité pour les recettes
)

# Créer l'agent avec mémoire et outils
agent = create_agent(
    model=model,
    tools=[rechercher_recette],
    system_prompt=system_prompt,
    checkpoint=InMemorySaver(),  # Pour la mémoire
)

# Configuration pour la mémoire (thread unique)
config = {"configurable": {"thread_id": "chef_session_1"}}

# ==================== MÉMOIRE DES INGRÉDIENTS ====================

# Simulons les ingrédients du réfrigérateur (l'agent les mémorisera)
ingredients_initiaux = [
    "poulet (2 filets)",
    "tomates (3)",
    "oignon (1)",
    "ail (2 gousses)",
    "crème fraîche",
    "pâtes (spaghetti)",
    "parmesan",
    "courgette (1)",
    "œufs (4)",
    "lait"
]

print("\n📋 LISTE DES INGRÉDIENTS DISPONIBLES :")
for ing in ingredients_initiaux:
    print(f"   - {ing}")

# ==================== DIALOGUE AVEC L'AGENT ====================

def parler_a_chef(message):
    """Fonction utilitaire pour interagir avec l'agent"""
    print(f"\n👤 Vous : {message}")
    response = agent.invoke(
        {"messages": [HumanMessage(content=message)]},
        config
    )
    reponse_chef = response['messages'][-1].content
    print(f"🍳 Chef : {reponse_chef}")
    return reponse_chef

print("\n" + "="*50)
print("👨‍🍳 DISCUSSIONS AVEC LE CHEF")
print("="*50)

# Étape 1 : Présentation et enregistrement des ingrédients
parler_a_chef(
    "Bonjour ! Je suis ravi de faire ta connaissance. "
    "Voici ce que j'ai dans mon réfrigérateur aujourd'hui : "
    "du poulet, des tomates, un oignon, de l'ail, de la crème fraîche, "
    "des spaghetti, du parmesan, une courgette, des œufs et du lait. "
    "Peux-tu me proposer un plat avec ces ingrédients ?"
)

# Étape 2 : Enregistrement des préférences (mémoire)
parler_a_chef(
    "Ah, et pour que tu saches : je n'aime pas trop les plats trop épicés, "
    "et je préfère cuisiner rapidement (moins de 30 minutes). "
    "Est-ce que tu as retenu ces informations ?"
)

# Étape 3 : Demander une nouvelle recommandation (vérifie la mémoire)
parler_a_chef(
    "Et si je veux utiliser uniquement les courgettes et les œufs que j'ai, "
    "qu'est-ce que tu me proposes ?"
)

# Étape 4 : Utilisation de la recherche web
parler_a_chef(
    "Je n'ai jamais cuisiné de sauce tomate maison. "
    "Peux-tu rechercher une bonne recette de sauce tomate rapide ?"
)

print("\n" + "="*50)
print("✅ Fin de la démonstration !")
print("Le chef a mémorisé vos préférences et ingrédients !")