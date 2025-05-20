import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
import plotly.graph_objects as go
from textblob import TextBlob
import nltk
from nltk.corpus import stopwords
import datetime
from collections import Counter
import re

# Initial setup
st.set_page_config(page_title="Social Media Sentiment Dashboard", layout="wide")
sns.set_style("whitegrid")

# Download necessary NLTK data - fixed the punkt resources
@st.cache_resource
def download_nltk_resources():
    try:
        nltk.data.find('tokenizers/punkt')
    except LookupError:
        nltk.download('punkt')

    try:
        nltk.data.find('corpora/stopwords')
    except LookupError:
        nltk.download('stopwords')

download_nltk_resources()

# Function to generate sample data
def generate_sample_data(n_samples=1000):
    # Generate sample data
    platforms = ['Twitter', 'Facebook', 'Instagram', 'Reddit', 'LinkedIn']
    topics = ['product', 'service', 'support', 'features', 'pricing', 'interface']
    sentiments = ['positive', 'negative', 'neutral']
    sentiment_words = {
        'positive': ['great', 'excellent', 'awesome', 'love', 'amazing', 'helpful', 'best', 'perfect', 'fantastic', 'recommend'],
        'negative': ['bad', 'terrible', 'awful', 'hate', 'horrible', 'useless', 'worst', 'disappointed', 'frustrating', 'avoid'],
        'neutral': ['okay', 'fine', 'average', 'decent', 'normal', 'standard', 'usual', 'common', 'regular', 'typical']
    }
    
    now = datetime.datetime.now()
    dates = [now - datetime.timedelta(minutes=np.random.randint(1, 60*24*7)) for _ in range(n_samples)]
    
    data = []
    for i in range(n_samples):
        platform = np.random.choice(platforms)
        user_id = f"user_{np.random.randint(1, 1000)}"
        sentiment = np.random.choice(sentiments, p=[0.4, 0.3, 0.3])
        topic = np.random.choice(topics)
        
        # Create post with biased sentiment words
        words = np.random.choice(sentiment_words[sentiment], size=np.random.randint(1, 4))
        post = f"I {words[0]} this {topic}. "
        if len(words) > 1:
            post += f"It's really {words[1]}. "
        if len(words) > 2:
            post += f"I would {words[2]} it to others."
            
        data.append([dates[i], platform, user_id, post])
    
    return pd.DataFrame(data, columns=['timestamp', 'platform', 'user_id', 'post'])

# Function to analyze sentiment
def analyze_sentiment(text):
    analysis = TextBlob(text)
    # Determine sentiment
    polarity = analysis.sentiment.polarity
    if polarity > 0.1:
        return 'Positive', polarity
    elif polarity < -0.1:
        return 'Negative', polarity
    else:
        return 'Neutral', polarity

# Function to extract common words
def extract_common_words(texts, sentiment_type=None, top_n=50):
    # Make sure stopwords are available
    stop_words = set(stopwords.words('english'))
    
    words = []
    
    for text in texts:
        # Clean text
        text = re.sub(r'[^\w\s]', '', text.lower())
        # Split into words - using simple split to avoid punkt issues
        tokens = text.split()
        # Remove stopwords and short words
        tokens = [word for word in tokens if word not in stop_words and len(word) > 2]
        words.extend(tokens)
    
    # Count word frequencies
    word_freq = Counter(words)
    return word_freq.most_common(top_n)

# Main app
def main():
    st.title("Social Media Sentiment Analysis Dashboard")
    
    # Make sure NLTK data is downloaded before proceeding
    with st.spinner("Setting up NLP resources..."):
        download_nltk_resources()
    
    # Load sample data
    data = generate_sample_data()
    
    # Add sentiment analysis
    sentiments = []
    polarity_scores = []
    
    for post in data['post']:
        sentiment, polarity = analyze_sentiment(post)
        sentiments.append(sentiment)
        polarity_scores.append(polarity)
    
    data['sentiment'] = sentiments
    data['polarity'] = polarity_scores
    
    # Add date column to avoid errors in filtering
    data['date'] = data['timestamp'].dt.date
    
    # Filter data by date range
    st.sidebar.header("Filters")
    
    # Date range filter
    min_date = data['date'].min()
    max_date = data['date'].max()
    
    start_date = st.sidebar.date_input("Start Date", min_date)
    end_date = st.sidebar.date_input("End Date", max_date)
    
    # Platform filter
    platforms = ['All'] + sorted(data['platform'].unique().tolist())
    selected_platform = st.sidebar.selectbox("Platform", platforms)
    
    # Sentiment filter
    sentiments = ['All'] + sorted(data['sentiment'].unique().tolist())
    selected_sentiment = st.sidebar.selectbox("Sentiment", sentiments)
    
    # Apply filters
    filtered_data = data.copy()
    
    # Date filter - fixed to use date column directly
    filtered_data = filtered_data[
        (filtered_data['date'] >= start_date) & 
        (filtered_data['date'] <= end_date)
    ]
    
    # Platform filter
    if selected_platform != 'All':
        filtered_data = filtered_data[filtered_data['platform'] == selected_platform]
    
    # Sentiment filter
    if selected_sentiment != 'All':
        filtered_data = filtered_data[filtered_data['sentiment'] == selected_sentiment]
    
    # Display filtered data stats
    st.sidebar.markdown(f"**Posts:** {len(filtered_data)}")
    
    # Dashboard layout
    # First row with two columns
    col1, col2 = st.columns(2)
    
    # 1. Overall Sentiment Distribution (Pie Chart)
    with col1:
        st.subheader("1. Overall Sentiment Distribution")
        sentiment_counts = filtered_data['sentiment'].value_counts().reset_index()
        sentiment_counts.columns = ['Sentiment', 'Count']
        
        fig = px.pie(
            sentiment_counts, 
            values='Count', 
            names='Sentiment',
            color='Sentiment',
            color_discrete_map={
                'Positive': '#4CAF50',
                'Neutral': '#FFC107',
                'Negative': '#F44336'
            },
            hole=0.4
        )
        fig.update_layout(height=400)
        st.plotly_chart(fig, use_container_width=True)
    
    # 2. Sentiment Over Time (Line Chart)
    with col2:
        st.subheader("2. Sentiment Trend Over Time")
        
        # Group by date and sentiment
        sentiment_by_date = filtered_data.groupby(['date', 'sentiment']).size().unstack().fillna(0)
        
        # Calculate percentage
        sentiment_by_date_pct = sentiment_by_date.div(sentiment_by_date.sum(axis=1), axis=0) * 100
        
        # Create line chart
        fig = go.Figure()
        
        if 'Positive' in sentiment_by_date_pct.columns:
            fig.add_trace(go.Scatter(
                x=sentiment_by_date_pct.index, 
                y=sentiment_by_date_pct['Positive'], 
                mode='lines+markers',
                name='Positive',
                line=dict(color='#4CAF50', width=3)
            ))
        
        if 'Neutral' in sentiment_by_date_pct.columns:
            fig.add_trace(go.Scatter(
                x=sentiment_by_date_pct.index, 
                y=sentiment_by_date_pct['Neutral'], 
                mode='lines+markers',
                name='Neutral',
                line=dict(color='#FFC107', width=3)
            ))
        
        if 'Negative' in sentiment_by_date_pct.columns:
            fig.add_trace(go.Scatter(
                x=sentiment_by_date_pct.index, 
                y=sentiment_by_date_pct['Negative'], 
                mode='lines+markers',
                name='Negative',
                line=dict(color='#F44336', width=3)
            ))
        
        fig.update_layout(
            xaxis_title='Date',
            yaxis_title='Percentage (%)',
            legend_title='Sentiment',
            height=400
        )
        st.plotly_chart(fig, use_container_width=True)

    # Second row with two columns
    col3, col4 = st.columns(2)
    
    # 3. Platform Comparison (Bar Chart)
    with col3:
        st.subheader("3. Sentiment by Platform")
        
        platform_sentiment = filtered_data.groupby(['platform', 'sentiment']).size().reset_index()
        platform_sentiment.columns = ['Platform', 'Sentiment', 'Count']
        
        fig = px.bar(
            platform_sentiment,
            x='Platform',
            y='Count',
            color='Sentiment',
            barmode='group',
            color_discrete_map={
                'Positive': '#4CAF50',
                'Neutral': '#FFC107',
                'Negative': '#F44336'
            }
        )
        fig.update_layout(height=400)
        st.plotly_chart(fig, use_container_width=True)
    
    # 4. Hourly Activity Heatmap
    with col4:
        st.subheader("4. Activity by Hour and Day")
        
        # Extract day and hour
        filtered_data['day'] = filtered_data['timestamp'].dt.day_name()
        filtered_data['hour'] = filtered_data['timestamp'].dt.hour
        
        # Create pivot table for heatmap
        hourly_activity = filtered_data.groupby(['day', 'hour']).size().reset_index()
        hourly_activity.columns = ['Day', 'Hour', 'Count']
        
        # Order days properly
        day_order = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday']
        hourly_activity['Day'] = pd.Categorical(hourly_activity['Day'], categories=day_order, ordered=True)
        hourly_activity = hourly_activity.sort_values(['Day', 'Hour'])
        
        # Create pivot table
        hourly_pivot = hourly_activity.pivot(index='Day', columns='Hour', values='Count').fillna(0)
        
        # Create heatmap
        fig = px.imshow(
            hourly_pivot,
            color_continuous_scale='Blues',
            labels=dict(x="Hour of Day", y="Day of Week", color="Post Count"),
            x=[f"{h}:00" for h in range(24)],
            y=day_order,
        )
        fig.update_layout(height=400)
        st.plotly_chart(fig, use_container_width=True)

    # Third row with three columns
    col5, col6, col7 = st.columns(3)
    
    # 5. Topic Sentiment Comparison (Replacing Radar Chart with a Grouped Bar Chart)
    with col5:
        st.subheader("5. Topic Sentiment Comparison")
        
        # Extract topics from posts
        topics = ['product', 'service', 'support', 'features', 'pricing', 'interface']
        
        # Calculate counts and average sentiment for each topic
        topic_data = []
        for topic in topics:
            # Filter posts containing this topic
            topic_posts = filtered_data[filtered_data['post'].str.contains(topic, case=False)]
            if not topic_posts.empty:
                # Calculate average polarity and count for this topic
                avg_polarity = topic_posts['polarity'].mean()
                count = len(topic_posts)
                topic_data.append({
                    'Topic': topic.capitalize(),
                    'Average Sentiment': avg_polarity,
                    'Post Count': count
                })
        
        if topic_data:
            # Create dataframe
            topic_df = pd.DataFrame(topic_data)
            
            # Create bar chart
            fig = px.bar(
                topic_df,
                x='Topic',
                y='Average Sentiment',
                color='Average Sentiment',
                color_continuous_scale='RdYlGn',
                range_color=[-1, 1],
                text='Post Count',
                hover_data=['Post Count'],
            )
            
            fig.add_hline(y=0, line_dash="dash", line_color="gray")
            fig.update_traces(texttemplate='%{text}', textposition='outside')
            fig.update_layout(height=400)
            
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.info("Not enough topic data with the selected filters")

    # 6. Polarity Distribution (Histogram)
    with col6:
        st.subheader("6. Sentiment Polarity Distribution")
        
        fig = px.histogram(
            filtered_data,
            x='polarity',
            nbins=50,
            color_discrete_sequence=['#1E88E5'],
            labels={'polarity': 'Sentiment Polarity', 'count': 'Frequency'}
        )
        
        fig.add_vline(x=0, line_dash="dash", line_color="gray")
        fig.update_layout(
            xaxis_title='Sentiment Polarity (Negative → Positive)',
            yaxis_title='Frequency',
            height=400
        )
        
        st.plotly_chart(fig, use_container_width=True)
    
    # 7. Top Words by Sentiment
    with col7:
        st.subheader("7. Top Words by Sentiment")
        
        # Radio buttons to select sentiment for top words
        selected_sentiment_words = st.radio(
            "Select sentiment",
            ['Positive', 'Negative', 'Neutral'],
            horizontal=True
        )
        
        # Get the posts for the selected sentiment
        sentiment_posts = filtered_data[filtered_data['sentiment'] == selected_sentiment_words]['post']
        
        if not sentiment_posts.empty:
            # Extract common words
            common_words = extract_common_words(sentiment_posts, top_n=10)
            
            # Create bar chart
            words_df = pd.DataFrame(common_words, columns=['Word', 'Frequency'])
            
            color_map = {
                'Positive': '#4CAF50',
                'Negative': '#F44336',
                'Neutral': '#FFC107'
            }
            
            fig = px.bar(
                words_df,
                y='Word',
                x='Frequency',
                orientation='h',
                color_discrete_sequence=[color_map[selected_sentiment_words]]
            )
            
            fig.update_layout(height=400)
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.info(f"No {selected_sentiment_words.lower()} posts found with current filters")

    # Add a new row with one column for the user volume trend
    st.subheader("User Volume and Engagement Trend")
    
    # Calculate daily post volume
    daily_volume = filtered_data.groupby('date').size().reset_index()
    daily_volume.columns = ['Date', 'Post Count']
    
    # Calculate daily unique users
    daily_users = filtered_data.groupby('date')['user_id'].nunique().reset_index()
    daily_users.columns = ['Date', 'Unique Users']
    
    # Merge the data
    engagement_data = pd.merge(daily_volume, daily_users, on='Date')
    
    # Calculate posts per user ratio
    engagement_data['Posts per User'] = engagement_data['Post Count'] / engagement_data['Unique Users']
    
    # Create the visualization
    fig = go.Figure()
    
    fig.add_trace(go.Bar(
        x=engagement_data['Date'],
        y=engagement_data['Post Count'],
        name='Total Posts',
        marker_color='#42A5F5'
    ))
    
    fig.add_trace(go.Bar(
        x=engagement_data['Date'],
        y=engagement_data['Unique Users'],
        name='Unique Users',
        marker_color='#66BB6A'
    ))
    
    fig.add_trace(go.Scatter(
        x=engagement_data['Date'],
        y=engagement_data['Posts per User'],
        name='Posts per User',
        mode='lines+markers',
        yaxis='y2',
        line=dict(color='#FFA726', width=3)
    ))
    
    fig.update_layout(
        barmode='group',
        xaxis_title='Date',
        yaxis_title='Count',
        yaxis2=dict(
            title='Posts per User',
            overlaying='y',
            side='right'
        ),
        legend=dict(
            orientation='h',
            yanchor='bottom',
            y=1.02,
            xanchor='right',
            x=1
        ),
        height=400
    )
    
    st.plotly_chart(fig, use_container_width=True)

    # Data table at the bottom
    st.subheader("Recent Posts")
    st.dataframe(
        filtered_data[['timestamp', 'platform', 'user_id', 'post', 'sentiment', 'polarity']]
        .sort_values('timestamp', ascending=False)
        .head(10)
    )

if __name__ == "__main__":
    main()