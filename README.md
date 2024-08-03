# The Probability Post: README

## Overview

The Probability Post is a Python-based project that generates a weekly newsletter summarizing the latest research papers in statistics from arXiv. The project uses N-shot prompting with the LLAMA 3 model to generate summaries and categorize abstracts into specific subfields of statistics. The newsletter includes a spotlight section, detailed summaries with categorizations, and a humorous joke.

## Project Structure

- **main**: The entry point of the program that coordinates the workflow.
- **fetch_latest_papers**: Retrieves the latest statistics papers from arXiv.
- **summarize_abstract**: Summarizes the abstract of a paper using the LLAMA 3 model.
- **categorize_abstract_with_llama**: Categorizes the summarized abstract into a predefined list of statistics subfields.
- **get_categorization_explanation**: Generates a two-sentence explanation for the categorization of each paper.
- **calculate_rouge**: Calculates the ROUGE score to evaluate the quality of the generated summary.
- **generate_category_paragraph**: Generates a paragraph discussing the significance and relation of the papers within each category.
- **generate_joke**: Generates a humorous joke related to statistics.
- **create_newsletter**: Combines all the components to create the final newsletter.

## Dependencies

- `arxiv`: To retrieve the latest research papers.
- `ollama`: To interact with the LLAMA 3 model for generating summaries, categorizing abstracts, and creating jokes.
- `rouge_score`: To calculate ROUGE scores for evaluating summaries.

## Setup Instructions

1. **Clone the Repository**: Clone the repository to your local machine using the following command:
   ```bash
   git clone <repository_url>
   ```

2. **Install Dependencies**: Install the required Python packages by running:
   ```bash
   pip install arxiv ollama rouge_score
   ```

3. **Run the Script**: Execute the main script to generate the newsletter:
   ```bash
   python main.py
   ```

## Functions and Usage

### Fetch Latest Papers


def fetch_latest_papers():
    print("Fetching results from arXiv...")
    client = arxiv.Client()
    search = arxiv.Search(
        query="cat:stat.*",
        max_results=15,
        sort_by=arxiv.SortCriterion.SubmittedDate
    )
    papers = list(client.results(search))
    print(f"Fetched {len(papers)} results.")
    return papers
```
- Retrieves the latest 15 statistics papers from arXiv.
- Returns a list of paper objects.

### Summarize Abstract


def summarize_abstract(abstract_text):
    print("Summarizing abstract...")
    try:
        response = ollama.chat(model="llama3", messages=[
            {
                'role': 'user',
                'content': f'Summarize the following abstract in one sentence: "{abstract_text}"'
            },
        ])
        if isinstance(response, dict) and 'message' in response and 'content' in response['message']:
            summary = response['message']['content']
            return summary.strip()
        else:
            return "Summary not available."
    except Exception as e:
        print(f"An error occurred while summarizing the text: {e}")
        return "Summary not available."
```
- Generates a one-sentence summary of the given abstract using the LLAMA 3 model.
- Returns the generated summary or an error message.

### Categorize Abstract


def categorize_abstract_with_llama(summary):
    print("Categorizing summary...")
    categories = [
        "Bayesian Statistics", "Computational Statistics", "Biostatistics",
        "Unsupervised Learning", "Supervised Learning",
        "High-Dimensional Statistics", "Time Series Analysis", "Multivariate Analysis",
        "Experimental Design", "Nonparametric Statistics", "Econometrics",
        "Probability Theory", "Statistical Learning Theory", "Applied Statistics",
        "Environmental Statistics", "Financial Statistics", "Survey Statistics",
        "Spatial Statistics", "Stochastic Processes", "Data Mining", "Neural Networks",
        "Reinforcement Learning", "Ensemble Learning", "Inferential Statistics",
        "Descriptive Statistics", "Machine Learning", "Statistics Sampling", "Bioinformatics",
        "Statistical Decision Theory", "Casual Inference", "Robust Statistics"
    ]
    try:
        response = ollama.chat(model="llama3", messages=[
            {
                'role': 'user',
                'content': f'Categorize the following summary into one of the given categories: "{summary}". Categories: {", ".join(categories)}.'
            },
        ])
        if isinstance(response, dict) and 'message' in response and 'content' in response['message']:
            category = response['message']['content'].strip()
            if any(cat.lower() in category.lower() for cat in categories):
                return next(cat for cat in categories if cat.lower() in category.lower())
            else:
                print(f"Received uncategorized response: {category}")
                return "Uncategorized"
        else:
            return "Uncategorized"
    except Exception as e:
        print(f"An error occurred while categorizing the summary: {e}")
        return "Uncategorized"
```
- Categorizes the summary into one of the predefined statistics subfields using the LLAMA 3 model.
- Returns the identified category or "Uncategorized."

### Categorization Explanation


def get_categorization_explanation(summary, category):
    print("Generating categorization explanation...")
    try:
        response = ollama.chat(model="llama3", messages=[
            {
                'role': 'user',
                'content': f"In two sentences explain why the following summary is categorized under '{category}': \"{summary}\""
            },
        ])
        if isinstance(response, dict) and 'message' in response and 'content' in response['message']:
            explanation = response['message']['content']
            return explanation.strip()
        else:
            return "Categorization explanation not available."
    except Exception as e:
        print(f"An error occurred while generating the categorization explanation: {e}")
        return "Categorization explanation not available."
```
- Generates a two-sentence explanation for why the summary is categorized under the given category.
- Returns the generated explanation or an error message.

### Calculate ROUGE Score


def calculate_rouge(reference, summary):
    scorer = rouge_scorer.RougeScorer(['rouge1', 'rougeL'], use_stemmer=True)
    scores = scorer.score(reference, summary)
    return scores
```
- Calculates the ROUGE score to evaluate the quality of the summary against the reference abstract.
- Returns the ROUGE scores.

### Generate Category Paragraph


def generate_category_paragraph(category, papers):
    try:
        prompt = f"Discuss how the following papers in the category '{category}' relate to each other and their collective significance:\n"
        for summary, paper in papers:
            authors = ', '.join(author.name for author in paper.authors)
            link = paper.entry_id
            prompt += f"Title: {paper.title}\nAuthors: {authors}\nSummary: {summary}\n\n"

        response = ollama.chat(model="llama3", messages=[
            {
                'role': 'user',
                'content': prompt
            },
        ])

        if isinstance(response, dict) and 'message' in response and 'content' in response['message']:
            category_paragraph = response['message']['content']
        else:
            category_paragraph = "Category discussion not available."

        return category_paragraph
    except Exception as e:
        print(f"An error occurred while generating category paragraph: {e}")
        return "Category discussion not available."
```
- Generates a paragraph discussing the significance and relation of papers within each category using the LLAMA 3 model.
- Returns the generated paragraph or an error message.

### Generate Joke


def generate_joke():
    try:
        examples = [
            "Chuck Norris can calculate the standard deviation of a dataset just by glaring at it. The data points are too afraid to stray far from the mean.",
            "Three statisticians walk into a bar. The bartender asks, 'What'll it be?' The first statistician says, 'I'll have a beer.' The second says, 'I'll have a beer too.' The third says, 'I'll abstain. Someone has to be the control group.'",
            "Why did the correlation coefficient go to therapy? It had a complex relationship with causation.",
            "I tried to conduct a survey about procrastination, but everyone said they'd fill it out later."
        ]
        example_prompts = "\n".join([f"Joke: {joke}" for joke in examples])

        response = ollama.chat(model="llama3", messages=[
            {
                'role': 'user',
                'content': f"Here are some examples of jokes:\n{example_prompts}\n\nNow, create a joke based on the information in the newsletter and title it 'The Punchline'."
            },
        ])

        if isinstance(response, dict) and 'message' in response and 'content' in response['message']:
            joke = response['message']['content']
        else:
            joke = "Joke creation failed."

        return joke
    except Exception as e:
        print(f"An error occurred while generating the joke: {e}")
        return "Joke creation failed."
```
- Generates a humorous joke related to statistics using the LLAMA 3 model.
- Returns the generated joke or an error message.

### Create Newsletter


def create_newsletter(categorized, summaries_and_categories):
    print("Creating the newsletter catered to statistics researchers and PhDs...")
    try

:
        prompt_header = "The Probability Post\n\nHi Stat Fam,\nWelcome to the latest edition of The Probability Post, where we bring you cutting-edge research from the world of statistics. Let's dive into the exciting new papers that are shaping our field!\n\n"

        spotlight_paper = summaries_and_categories[0][2]
        spotlight_summary, spotlight_category = summaries_and_categories[0][0], summaries_and_categories[0][1]
        spotlight_prompt = (
            f"Title: {spotlight_paper.title}\n"
            f"Authors: {', '.join(author.name for author in spotlight_paper.authors)}\n"
            f"Summary: {spotlight_summary}\n"
            f"Link: {spotlight_paper.entry_id}\n\n"
            f"Explanation: This paper is highlighted for its innovative approach to handling data heterogeneity in federated learning. By improving patient-centric personalized survival analysis, it addresses a critical challenge in medical data analysis, ensuring that models are robust across diverse datasets. The proposed federated learning approach mitigates data heterogeneity and enhances model performance across synthetic and real-world applications, including cancer data. This methodology not only improves the accuracy of survival predictions but also ensures that personalized medicine can be more effectively implemented, leading to better patient outcomes. The integration of federated learning into survival analysis represents a major advancement in the field, with potential applications in various domains where data privacy and heterogeneity are concerns."
        )

        newsletter_sections = []
        for category, summaries in categorized.items():
            if summaries:
                category_paragraph = generate_category_paragraph(category, summaries)
                section = f"\n\n{category}\n{category_paragraph}\n"
                for i, (summary, paper) in enumerate(summaries, start=1):
                    authors = ', '.join(author.name for author in paper.authors)
                    link = paper.entry_id
                    explanation = get_categorization_explanation(summary, category)
                    section += f"\n{i}. {paper.title}\nAuthors: {authors}\nSummary: {summary}\nLink: [Link]({link})\nCategorization Explanation: {explanation}\n"
                newsletter_sections.append(section)

        newsletter_body = "\n\n".join(newsletter_sections)

        joke = generate_joke()

        sign_off = "\n\nThat's all for this edition of The Probability Post! Stay tuned for more cutting-edge research and statistical insights. See you next week!\n\nBest regards,\nThe Probability Post Team"

        newsletter = f"{prompt_header}\nThe Spotlight\n\n{spotlight_prompt}\n\n{newsletter_body}\n\nThe Punchline\n\n{joke}\n{sign_off}"

        return newsletter
    except Exception as e:
        print(f"An error occurred while creating the newsletter: {e}")
        return "Newsletter creation failed."
```
- Combines all components to create the final newsletter.
- Returns the generated newsletter or an error message.

## Running the Project

1. Ensure all dependencies are installed.
2. Run the main script:
   ```bash
   python main.py
   ```
3. The script will fetch the latest papers, generate summaries and categorizations, and compile the final newsletter.

## Customization

- **Number of Papers**: Adjust the `max_results` parameter in the `fetch_latest_papers` function to change the number of papers retrieved from arXiv.
- **Categories**: Modify the list of categories in the `categorize_abstract_with_llama` function to include additional or different subfields of statistics.

