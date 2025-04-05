from setuptools import setup, find_packages

setup(
    name="mcp_integrator",
    version="0.1.0",
    packages=find_packages(),
    install_requires=[
        "requests",
        "pydantic",
        "openai",
        "python-dotenv",
    ],
    entry_points={
        'console_scripts': [
            'mcp-finder=mcp_integrator.cli.mcp_finder_cli:main',
            'stack-recommender=mcp_integrator.cli.stack_recommender_cli:main',
        ],
    },
    author="Hydra Team",
    author_email="team@hydra.ai",
    description="MCP Integration Agent for Hydra",
    long_description=open("README.md").read(),
    long_description_content_type="text/markdown",
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    python_requires=">=3.8",
) 