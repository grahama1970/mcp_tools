from setuptools import setup, find_packages

setup(
    name="claude-task-manager",
    version="1.0.0",
    packages=find_packages(),
    install_requires=[],
    entry_points={
        'console_scripts': [
            'claude-tasks=claude_task_manager.cli:main',
        ],
    },
    author="Robert",
    author_email="",
    description="Task management system for Claude Code with context isolation",
    long_description=open("README.md").read(),
    long_description_content_type="text/markdown",
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    python_requires=">=3.7",
)
