from flask import Flask, request, render_template, session
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import neattext.functions as nfx
import os
from dotenv import load_dotenv
import google.generativeai as genai
import markdown

load_dotenv()

# Configure Gemini

model = genai.GenerativeModel('gemini-2.5-flash-lite')

app = Flask(__name__)
app.secret_key = os.getenv('SECRET_KEY', 'default_secret_key')

# Load the dataset
data = pd.read_csv('udemy.csv')

# Clean the data: remove rows with NaN in course_title, subject, or level
data = data.dropna(subset=['course_title', 'subject', 'level'])

# Create Clean_title by cleaning course_title
data['Clean_title'] = data['course_title'].apply(nfx.remove_stopwords)
data['Clean_title'] = data['Clean_title'].apply(nfx.remove_punctuations)

# Ensure price is numeric and convert from EGP to NGN (1 EGP = 30.19 NGN)
data['price'] = pd.to_numeric(data['price'], errors='coerce').fillna(0) * 30.19

# Combine relevant text for vectorization: title, subject, level
data['combined_text'] = data['Clean_title'] + ' ' + data['subject'] + ' ' + data['level']

# Initialize TfidfVectorizer for better weighting
vectorizer = TfidfVectorizer(stop_words='english')
tfidf_matrix = vectorizer.fit_transform(data['combined_text'])

def handle_ai_agent(user_input, session):
    step = session.get('step', 0)
    profile = session.get('profile', {})

    if step == 0:
        # Initial greeting and ask for learning goal
        session['step'] = 1
        return "Hi! I'm your AI Learning Advisor. What would you like to learn? For example, 'Python programming', 'data science', or 'web development'."

    elif step == 1:
        # Store learning goal
        profile['goal'] = user_input
        session['step'] = 2
        return f"Awesome! Learning {user_input} is a great goal. What's your current skill level? (Beginner, Intermediate, Advanced)"

    elif step == 2:
        # Store skill level
        profile['skill_level'] = user_input.lower()
        session['step'] = 3
        return f"Got it, you're at {user_input} level. What are your learning goals? (career change, personal interest, academic requirement)"

    elif step == 3:
        # Store goals
        profile['learning_goals'] = user_input
        session['step'] = 4
        return "How much time can you dedicate per week? (e.g., 5-10 hours, 10-20 hours, more than 20)"

    elif step == 4:
        # Store time availability
        profile['time_availability'] = user_input
        session['step'] = 5
        return "Do you prefer practical, hands-on learning, theoretical concepts, or a mix of both?"

    elif step == 5:
        # Store preferences
        pref = user_input.lower()
        if pref == 'mix':
            pref = 'mixed'
        profile['preferences'] = pref
        session['ready_to_recommend'] = True
        session['show_results'] = True
        session['step'] = 6  # Move to follow-up
        return "Thanks! I've analyzed your learning profile and selected courses that best match your goal, level, and available time. Here are your personalized recommendations."

    elif step == 6:
        # Follow-up conversation
        short_responses = ['no', 'yes', 'okay', 'ok', 'thanks', 'thank you', 'good', 'fine']
        if len(user_input.split()) < 2 and user_input.lower() in short_responses:
            return "Is there a specific course you'd like more details on, or would you like recommendations in another area?"
        course = data[data['course_title'].str.contains(user_input, case=False, na=False)]
        if not course.empty:
            # User mentioned a course, provide details
            course = course.iloc[0]
            return f"Here's more info on '{course['course_title']}':\n- Subject: {course['subject']}\n- Level: {course['level']}\n- Price: ₦{course['price']:.0f}\n- Rating: {course['Rating']}\n- Duration: {course['content_duration']} hours\n- URL: {course['url']}"
        else:
            return "Is there a specific course you'd like more details on, or would you like recommendations in another area?"

    return "I'm ready to help you find the perfect courses!"

def get_ai_recommendations(profile):
    query = profile.get('goal', '')
    filters = {}

    # Adjust filters based on profile
    skill = profile.get('skill_level', '').lower()
    if 'beginner' in skill:
        filters['level'] = 'Beginner Level'
    elif 'intermediate' in skill:
        filters['level'] = 'Intermediate Level'
    elif 'advanced' in skill:
        filters['level'] = 'Expert Level'

    # Get initial recommendations
    recommendations = get_recommendations(query, filters, top_n=10)

    # Overload prevention
    time_avail = profile.get('time_availability', '').lower()
    if 'beginner' in skill and ('5-10' in time_avail or 'less' in time_avail):
        # Filter out courses with duration > 20 hours or too expensive
        recommendations = recommendations[recommendations['content_duration'].fillna(0) <= 20]
        recommendations = recommendations[recommendations['price'] <= 10000]  # Arbitrary limit

    # AI reasoning for ranking
    if 'beginner' in skill:
        recommendations = recommendations.sort_values(by=['Rating', 'similarity_score'], ascending=[False, False])
    else:
        recommendations = recommendations.sort_values(by='similarity_score', ascending=False)

    return recommendations.head(5)

def generate_explanations(recommendations, profile):
    explanations = []
    for _, rec in recommendations.iterrows():
        try:
            prompt = f"""
            Explain why this course "{rec['course_title']}" is suitable for a learner with:
            - Goal: {profile.get('goal', '')}
            - Skill level: {profile.get('skill_level', '')}
            - Learning goals: {profile.get('learning_goals', '')}
            - Time availability: {profile.get('time_availability', '')}
            - Preferences: {profile.get('preferences', '')}

            Provide a brief explanation in 2-3 bullet points, focusing on how it matches their profile.
            """
            response = model.generate_content(prompt)
            explanations.append(response.text.strip())
        except Exception as e:
            # Fallback explanation if API fails
            fallback = f"- Matches your {profile.get('goal', 'learning')} goal\n- Suitable for {profile.get('skill_level', 'your')} level\n- Aligns with your {profile.get('learning_goals', 'goals')}"
            explanations.append(fallback)
    return explanations

def generate_learning_path(recommendations, profile):
    try:
        courses_list = "\n".join([f"- {rec['course_title']} ({rec['level']})" for _, rec in recommendations.iterrows()])
        prompt = f"""
        Based on the learner's profile:
        - Goal: {profile.get('goal', '')}
        - Skill level: {profile.get('skill_level', '')}
        - Learning goals: {profile.get('learning_goals', '')}
        - Time availability: {profile.get('time_availability', '')}
        - Preferences: {profile.get('preferences', '')}

        And these recommended courses:
        {courses_list}

        Suggest a learning path: order the courses logically, and provide guidance on how to proceed (e.g., start with basics, practice, etc.). Keep it concise, 4-6 bullet points.
        """
        response = model.generate_content(prompt)
        return response.text.strip()
    except Exception as e:
        # Fallback learning path
        return "- Start with the highest-rated beginner course if you're new\n- Move to intermediate topics after completing the first\n- Practice regularly and apply what you learn\n- Consider projects or certifications to solidify your skills"





def get_recommendations(query, filters=None, top_n=10):
    """
    Recommends courses similar to the given query based on cosine similarity, with optional filters.

    Parameters:
    query (str): The search query.
    filters (dict): Optional filters (subject, level, max_price, min_rating).
    top_n (int): Number of recommendations to return.

    Returns:
    DataFrame: A DataFrame with recommended courses and their similarity scores.
    """
    if filters is None:
        filters = {}

    # Clean the query
    cleaned_query = nfx.remove_stopwords(query)
    cleaned_query = nfx.remove_punctuations(cleaned_query)

    # Vectorize the query
    query_vector = vectorizer.transform([cleaned_query])

    # Compute cosine similarities
    similarities = cosine_similarity(query_vector, tfidf_matrix).flatten()

    # Get top similar courses
    top_indices = similarities.argsort()[-top_n:][::-1]

    # Get recommendations
    recommendations = data.iloc[top_indices][['course_title', 'url', 'subject', 'level', 'price', 'Rating', 'content_duration', 'Clean_title']].copy()
    recommendations['similarity_score'] = similarities[top_indices]

    # Apply filters
    if 'level' in filters and filters['level']:
        recommendations = recommendations[recommendations['level'] == filters['level']]
    if 'max_price' in filters and filters['max_price'] is not None:
        recommendations = recommendations[recommendations['price'] <= filters['max_price']]
    if 'min_rating' in filters and filters['min_rating'] is not None:
        recommendations = recommendations[recommendations['Rating'] >= filters['min_rating']]

    return recommendations











@app.route('/', methods=['GET'])
def index():
    return render_template('index.html')

@app.route('/advisor', methods=['GET', 'POST'])
def advisor():
    if 'messages' not in session:
        session['messages'] = []
        session['profile'] = {}
        session['step'] = 0

    if request.args.get('back') == '1':
        session['show_results'] = False
        session.modified = True
    elif request.args.get('reset') == '1':
        session['messages'] = []
        session['profile'] = {}
        session['step'] = 0
        session['ready_to_recommend'] = False
        session['show_results'] = False
        session.modified = True

    if request.method == 'POST':
        user_input = request.form['user_input']
        session['messages'].append({'role': 'user', 'text': user_input})

        # AI Agent Logic
        response = handle_ai_agent(user_input, session)

        session['messages'].append({'role': 'ai', 'text': response})

        # Check if ready to recommend
        if session.get('ready_to_recommend', False):
            # Get recommendations based on profile
            recommendations = get_ai_recommendations(session['profile'])
            explanations = generate_explanations(recommendations, session['profile'])
            learning_path = generate_learning_path(recommendations, session['profile'])
            rec_exp = list(zip(recommendations.to_dict('records'), explanations))
            session['ready_to_recommend'] = False  # Prevent re-showing on back
            return render_template('ai_results.html', rec_exp=rec_exp, profile=session['profile'], learning_path=learning_path)

        session.modified = True

    # If ready to show results
    if session.get('show_results', False):
        recommendations = get_ai_recommendations(session['profile'])
        explanations = generate_explanations(recommendations, session['profile'])
        learning_path = generate_learning_path(recommendations, session['profile'])
        rec_exp = list(zip(recommendations.to_dict('records'), explanations))
        session['ready_to_recommend'] = False
        return render_template('ai_results.html', rec_exp=rec_exp, profile=session['profile'], learning_path=learning_path)

    return render_template('advisor.html', messages=session['messages'])

@app.route('/tutor', methods=['GET', 'POST'])
def tutor():
    if 'chat_messages' not in session:
        session['chat_messages'] = []

    if request.args.get('reset') == '1':
        session['chat_messages'] = []
        session.modified = True

    if request.method == 'POST':
        user_question = request.form['user_question']
        session['chat_messages'].append({'role': 'user', 'text': user_question})

        # Generate response using LLM
        response = handle_chatbot_question(user_question)
        session['chat_messages'].append({'role': 'ai', 'text': response})

        session.modified = True

    return render_template('tutor.html', messages=session['chat_messages'])

def handle_chatbot_question(question):
    try:
        # Create context from dataset: subjects and courses with URLs
        subjects = data['subject'].unique().tolist()
        # Get top 50 courses with titles and URLs
        sample_courses = data[['course_title', 'url']].head(50).to_dict('records')
        context = f"Available course subjects: {', '.join(subjects)}.\nCourses: " + "\n".join([f"- {c['course_title']}: {c['url']}" for c in sample_courses])

        prompt = f"""
        You are an intelligent chatbot tutor for a Udemy course recommendation system.
        Use the provided context to answer questions about courses and learning topics.
        Structure your responses clearly with sections and bullet points to avoid overloading.
        Use markdown formatting for better readability.

        Response Guidelines:
        1. **Explanation Section**: Provide clear, concise explanations of concepts
        2. **Course Recommendations**: If relevant, suggest 3-5 courses from the context
        3. **Examples**: Give targeted examples when appropriate
        4. **Links**: Always include course URLs in recommendations so users can click them

        Format recommendations like:
        **Recommended Courses:**
        - [Course Title](URL)
        - [Course Title](URL)

        Keep responses engaging but not overwhelming. Break into logical sections.

        Context: {context}

        User question: {question}
        """
        response = model.generate_content(prompt)
        # Convert markdown to HTML for better formatting
        html_response = markdown.markdown(response.text.strip(), extensions=['extra'])
        return html_response
    except Exception as e:
        return "Sorry, I'm having trouble answering that right now. Please try again later."

@app.route('/search', methods=['POST'])
def search():
    query = request.form['query']
    filters = {
        'level': request.form.get('level'),
        'max_price': float(request.form.get('max_price')) if request.form.get('max_price') else None,
        'min_rating': float(request.form.get('min_rating')) if request.form.get('min_rating') else None,
    }
    recommendations = get_recommendations(query, filters)
    return render_template('results.html', query=query, recommendations=recommendations.to_dict(orient='records'))









if __name__ == '__main__':

    app.run(debug=True)
