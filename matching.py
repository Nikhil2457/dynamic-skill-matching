import pandas as pd
import string

def simple_tokenize(text):
    """Tokenize using only built-in string methods"""
    text = text.lower().strip()
    for punct in string.punctuation:
        text = text.replace(punct, '')
    return text.split()

def extract_skills(text):
    """Improved skill extraction without NLTK"""
    text = text.lower().strip()
    skills = []
    for part in text.split(','):
        skills.extend(part.split(' and '))
    return set(word.strip() for word in simple_tokenize(' '.join(skills)) if word.strip())

def match_skills(users, project):
    """Match user skills to project requirements"""
    required_skills = extract_skills(project['requirements'])
    
    if not required_skills:
        return pd.DataFrame()

    def calculate_score(user_skills):
        user_skills_set = extract_skills(user_skills)
        return len(user_skills_set & required_skills) / len(required_skills) if required_skills else 0
    
    users['match_score'] = users['skills'].apply(calculate_score)
    sorted_users = users.sort_values(by=['match_score', 'experience'], ascending=[False, False]).reset_index(drop=True)
    sorted_users = sorted_users[sorted_users['match_score'] > 0]
    
    return sorted_users
