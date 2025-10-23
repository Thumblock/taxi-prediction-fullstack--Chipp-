from setuptools import setup, find_packages

setup(
    name="taxipred",
    version="0.1.0",
    description="Taxi price prediction package",
    author="Chipp",
    package_dir={"": "src"},
    packages=find_packages(where="src"),
    install_requires=[
        "pandas",
        "numpy",
        "scikit-learn",
        "matplotlib",
        "joblib",
        "fastapi",
        "uvicorn[standard]",
        "streamlit",
        "pydantic",
        "requests",
    ],
    package_data={"taxipred": ["data/*.csv"]},
    python_requires=">=3.9",
)

## ToDoList
# Add version floor constraints using the specifier >=