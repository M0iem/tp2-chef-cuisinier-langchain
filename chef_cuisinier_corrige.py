"""
TP : Chef Cuisinier Personnel - Version CORRIGÉE
Le paramètre correct est 'checkpointer' (pas 'checkpoint')
"""

from langchain.agents import create_agent
from langchain.messages import HumanMessage
from langchain_ollama import ChatOllama
from langgraph.checkpoint.memory import InMemorySaver
from langchain.tools import tool

# ==================== SIMULATION DE RECHERCHE WEB ====================

@tool("recherche_recette")
def rechercher_recette(requete: str) -> str:
    """
    Recherche des recettes ou techniques culinaires.
    """
    print(f"\n🔍 Recherche culinaire : '{requete}'")
    
    # Base de connaissances intégrée
    recettes_fixes = {
        "sauce tomate": """
🍅 RECETTE SAUCE TOMATE RAPIDE (15 min):
Ingrédients: tomates, oignon, ail, huile d'olive, sel, basilic
Étapes:
1. Émincer l'oignon et l'ail
2. Les faire revenir dans l'huile
3. Ajouter les tomates coupées
4. Laisser mijoter 10 min
5. Mixer et ajouter le basilic
""",
        "poulet": """
🍗 POULET À LA CRÈME (25 min):
Ingrédients: poulet, oignon, ail, crème fraîche, champignons
Étapes:
1. Dorer les morceaux de poulet
2. Ajouter oignon et ail
3. Verser la crème et mijoter 15 min
""",
        "omelette": """
🍳 OMELETTE PARFAITE (10 min):
Ingrédients: œufs, sel, poivre, option: fromage/herbes
Étapes:
1. Battre les œufs en omelette
2. Verser dans poêle chaude
3. Cuire à feu doux 3-4 min
4. Replier et servir
""",
        "courgette": """
🥒 COURGETTES SAUTÉES (10 min):
Ingrédients: courgette, ail, huile d'olive, sel, herbes
Étapes:
1. Laver et couper la courgette en rondelles
2. Faire chauffer l'huile dans une poêle
3. Ajouter l'ail émincé et les courgettes
4. Faire sauter 5-7 min à feu vif
5. Saler et ajouter des herbes
"""
    }
    
    # Chercher la recette qui correspond le mieux
    for mot_cle, recette in recettes_fixes.items():
        if mot_cle in requete.lower():
            return recette
    
    return f"""
📚 Idées pour "{requete}" :
• Recette classique: chercher "recette {requete} facile" sur internet
• Astuce: commencer avec des ingrédients simples
• Technique: maîtriser les bases avant les plats complexes
"""

# ==================== SYSTÈME PROMPT ====================

system_prompt = """Tu es un chef cuisinier personnel professionnel, chaleureux et expert en cuisine.

RÈGLES IMPORTANTES :
1. Mémorise toutes les informations que l'utilisateur te donne (préférences, ingrédients disponibles)
2. Propose des recettes adaptées aux ingrédients donnés
3. Utilise l'outil 'recherche_recette' pour trouver des recettes spécifiques
4. Propose toujours 1 à 2 plats avec les étapes clés
5. Sois enthousiaste et encourageant !
6. N'utilise PAS d'épices piquantes si l'utilisateur n'aime pas ça

Ton rôle : Aider à cuisiner délicieusement avec ce qu'on a sous la main !"""

# ==================== INITIALISATION ====================

print("🍳 Initialisation du Chef Cuisinier Personnel...")
print("="*50)

# Vérifier que Ollama est accessible
try:
    model = ChatOllama(
        model="llama3.2:3b",
        temperature=0.7,
    )
    print("✓ Modèle Ollama chargé avec succès")
except Exception as e:
    print(f"✗ Erreur Ollama: {e}")
    print("Assurez-vous qu'Ollama est installé et démarré!")
    exit(1)

# Créer le checkpointer pour la mémoire
checkpointer = InMemorySaver()
print("✓ Checkpointer (mémoire) initialisé")

# Créer l'agent - NOTE: le paramètre est 'checkpointer' (pas 'checkpoint')
agent = create_agent(
    model=model,
    tools=[rechercher_recette],
    system_prompt=system_prompt,
    checkpointer=checkpointer,  # ← Correction ici !
)
print("✓ Agent créé avec succès")

# Configuration pour la mémoire (thread_id unique pour cette conversation)
config = {"configurable": {"thread_id": "chef_session_1"}}

print("\n📋 INGRÉDIENTS DISPONIBLES DANS VOTRE FRIGO :")
ingredients = [
    "poulet (2 filets)",
    "tomates (3)",
    "oignon (1)",
    "ail (2 gousses)", 
    "crème fraîche",
    "spaghetti",
    "parmesan",
    "courgette (1)",
    "œufs (4)",
    "lait"
]
for ing in ingredients:
    print(f"   • {ing}")

# ==================== FONCTION DE DIALOGUE ====================

def discuter_avec_chef(message):
    """Dialogue avec l'agent"""
    print(f"\n👤 VOUS : {message}")
    print("🍳 CHEF : ", end="", flush=True)
    
    response = agent.invoke(
        {"messages": [HumanMessage(content=message)]},
        config
    )
    reponse = response['messages'][-1].content
    print(reponse)
    return reponse

print("\n" + "="*50)
print("👨‍🍳 COMMENÇONS LA CONVERSATION !")
print("="*50)

# Étape 1: Présentation des ingrédients
discuter_avec_chef(
    "Bonjour chef ! Voici mes ingrédients : poulet, tomates, oignon, "
    "ail, crème fraîche, spaghetti, parmesan, courgette, œufs et lait. "
    "Que puis-je cuisiner avec ça ?"
)

# Étape 2: Préférences personnelles (test de mémoire)
discuter_avec_chef(
    "Ah j'oubliais de préciser : je n'aime pas du tout les plats épicés, "
    "et je veux cuisiner rapidement en moins de 30 minutes. "
    "Tu as bien noté ça ?"
)

# Étape 3: Demander une omelette aux courgettes (vérifie mémoire)
discuter_avec_chef(
    "Si je veux utiliser uniquement les courgettes et les œufs, "
    "tu me proposes quoi comme plat rapide ?"
)

# Étape 4: Demander une recherche de recette
discuter_avec_chef(
    "Je ne sais pas faire une bonne sauce tomate maison. "
    "Peux-tu rechercher une recette de sauce tomate ?"
)

print("\n" + "="*50)
print("✅ FIN DE LA DÉMONSTRATION")
print("Le chef a mémorisé vos préférences !")
print("="*50)