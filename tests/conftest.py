"""Shared pytest fixtures for the course recommender test suite."""

import pandas as pd
import pytest


@pytest.fixture
def sample_raw_df() -> pd.DataFrame:
    """A small raw DataFrame mimicking the Coursera unclean dataset."""
    return pd.DataFrame(
        {
            "Course Name": [
                "Python for Everybody",
                "Machine Learning",
                "Python for Everybody",  # duplicate
                "Data Science Basics",
                None,  # missing critical field
                "Deep Learning Specialization",
                "SQL for Data Analysis",
            ],
            "University": [
                "University of Michigan",
                "Stanford University",
                "University of Michigan",  # duplicate
                "IBM",
                "MIT",
                "deeplearning.ai",
                "Duke University",
            ],
            "Difficulty Level": [
                "Beginner",
                "Intermediate",
                "Beginner",
                "beginner",  # inconsistent case
                "Advanced",
                "Advanced",
                "beginer",  # typo
            ],
            "Course Rating": [4.8, 4.9, 4.8, 4.5, 6.0, 4.7, 4.6],  # 6.0 is invalid
            "Course URL": [
                "https://coursera.org/python-everybody",
                "https://coursera.org/ml-stanford",
                "https://coursera.org/python-everybody",  # duplicate URL
                "https://coursera.org/ds-basics",
                "https://coursera.org/missing-name",
                "https://coursera.org/deep-learning",
                "https://coursera.org/sql-duke",
            ],
            "Course Description": [
                "Learn Python programming from scratch.",
                "Introduction to machine learning algorithms.",
                "Learn Python programming from scratch.",
                "Fundamentals of data science.",
                "Advanced research methods.",
                "Deep learning with TensorFlow and Keras.",
                "  SQL queries and database management.  ",  # extra whitespace
            ],
            "Skills": [
                "Python, Programming",
                "Machine Learning, Statistics",
                "Python, Programming",
                "Data Analysis, Python",
                "",
                "Deep Learning, Neural Networks, TensorFlow",
                "SQL, Databases",
            ],
        }
    )


@pytest.fixture
def sample_cleaned_df() -> pd.DataFrame:
    """A small cleaned DataFrame ready for model training."""
    return pd.DataFrame(
        {
            "course_id": [0, 1, 2, 3],
            "Course Name": [
                "Python for Everybody",
                "Machine Learning",
                "Data Science Basics",
                "Deep Learning Specialization",
            ],
            "Difficulty Level": ["Beginner", "Intermediate", "Beginner", "Advanced"],
            "Course Rating": [4.8, 4.9, 4.5, 4.7],
            "Course URL": [
                "https://coursera.org/python-everybody",
                "https://coursera.org/ml-stanford",
                "https://coursera.org/ds-basics",
                "https://coursera.org/deep-learning",
            ],
            "Course Description": [
                "Learn Python programming from scratch.",
                "Introduction to machine learning algorithms.",
                "Fundamentals of data science.",
                "Deep learning with TensorFlow and Keras.",
            ],
            "Skills": [
                "Python, Programming",
                "Machine Learning, Statistics",
                "Data Analysis, Python",
                "Deep Learning, Neural Networks, TensorFlow",
            ],
            "combined_text": [
                "Python for Everybody Learn Python programming from scratch. Python, Programming Beginner",
                "Machine Learning Introduction to machine learning algorithms. Machine Learning, Statistics Intermediate",
                "Data Science Basics Fundamentals of data science. Data Analysis, Python Beginner",
                "Deep Learning Specialization Deep learning with TensorFlow and Keras. Deep Learning, Neural Networks, TensorFlow Advanced",
            ],
            "has_rating": [True, True, True, True],
            "num_skills": [2, 2, 2, 3],
            "text_length": [38, 46, 29, 43],
        }
    )
