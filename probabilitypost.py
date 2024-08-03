import arxiv
import ollama
from rouge_score import rouge_scorer

# Function to retrieve the latest statistics papers
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

# N-shot prompting function for summarization and categorization
def summarize_abstract(abstract_text, prompt):
    print(f"Generating summary with prompt: {prompt}...")
    try:
        examples = [
            {
                "abstract": "We introduce a rigorous mathematical framework for Granger causality in extremes, designed to identify causal links from extreme events in time series...",
                "summary": "This paper introduces a mathematical framework for Granger causality in extremes, designed to identify causal relationships between extreme events in time series, offering advantages over traditional methods and demonstrating effectiveness in financial and extreme weather applications.",
                "category": "Extreme Value Theory"
            },
            {
                "abstract": "Due to the high dimensionality or multimodality that is common in modern astronomy, sampling Bayesian posteriors can be challenging...",
                "summary": "This paper describes a new, efficient C-language code called Nii-C that uses automatic parallel tempering and parallelization to improve sampling of complex probability distributions in astronomy and other fields, addressing challenges in high-dimensional or multimodal data analysis.",
                "category": "Computational Statistics"
            },
            {
                "abstract": "For a sequence of  n  random variables taking values 0 or 1, the hot hand statistic of streak length  k  counts what fraction of the streaks of length  k , that is,  k  consecutive variables taking the value 1, among the  n  variables are followed by another 1...",
                "summary": "The paper discusses a statistical measure called the 'hot hand statistic' that examines patterns in binary sequences, highlighting potential bias in estimating probabilities and proposing a new approach to calculate its expected value for single-event streaks.",
                "category": "Time Series Analysis"
            }
        ]
        example_prompts = "\n".join([
            f"Abstract: {ex['abstract']}\nSummary: {ex['summary']}\nCategory: {ex['category']}"
            for ex in examples
        ])

        response = ollama.chat(model="llama3", messages=[
            {
                'role': 'user',
                'content': f"{prompt}\n\nHere are some examples of summarizing and categorizing abstracts:\n{example_prompts}\n\nNow, summarize the following abstract in one sentence and provide the category: \"{abstract_text}\"."
            },
        ])
        if isinstance(response, dict) and 'message' in response and 'content' in response['message']:
            summary_category = response['message']['content'].strip()
            summary, category = summary_category.split("Category:")
            return summary.strip(), category.strip()
        return "Summary not available.", "Uncategorized"
    except Exception as e:
        print(f"An error occurred while summarizing the text: {e}")
        return "Summary not available.", "Uncategorized"

# Function to prompt Llama 3 for summarization
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

# Function to categorize text with Llama
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
            # Check if the response category is in the predefined list of categories
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

# Function to get categorization explanation
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

# Function to calculate ROUGE score
def calculate_rouge(reference, summary):
    scorer = rouge_scorer.RougeScorer(['rouge1', 'rougeL'], use_stemmer=True)
    scores = scorer.score(reference, summary)
    return scores

# Function to generate a paragraph discussing the papers in each category
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

# Function to generate a joke using N-shot prompting
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

# Function to create the newsletter
def create_newsletter(categorized, summaries_and_categories):
    print("Creating the newsletter catered to statistics researchers and PhDs...")
    try:
        prompt_header = "The Probability Post\n\nHi Stat Fam,\nWelcome to the latest edition of The Probability Post, where we bring you cutting-edge research from the world of statistics. Let's dive into the exciting new papers that are shaping our field!\n\n"

        # Generate the spotlight section
        spotlight_paper = summaries_and_categories[0][2]  # Choose the first paper as the spotlight for simplicity
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
            if summaries:  # Only add non-empty categories
                category_paragraph = generate_category_paragraph(category, summaries)
                section = f"\n\n{category}\n{category_paragraph}\n"
                for i, (summary, paper) in enumerate(summaries, start=1):
                    authors = ', '.join(author.name for author in paper.authors)
                    link = paper.entry_id
                    explanation = get_categorization_explanation(summary, category)
                    section += f"\n{i}. {paper.title}\nAuthors: {authors}\nSummary: {summary}\nLink: [Link]({link})\nCategorization Explanation: {explanation}\n"
                newsletter_sections.append(section)

        # Combine all sections
        newsletter_body = "\n\n".join(newsletter_sections)

        # Generate the joke
        joke = generate_joke()

        # Add the sign-off message
        sign_off = "\n\nThat's all for this edition of The Probability Post! Stay tuned for more cutting-edge research and statistical insights. See you next week!\n\nBest regards,\nThe Probability Post Team"

        # Combine everything into the final newsletter
        newsletter = f"{prompt_header}\nThe Spotlight\n\n{spotlight_prompt}\n\n{newsletter_body}\n\nThe Punchline\n\n{joke}\n{sign_off}"

        return newsletter
    except Exception as e:
        print(f"An error occurred while creating the newsletter: {e}")
        return "Newsletter creation failed."

# Main function
def main():
    try:
        papers = fetch_latest_papers()
        summaries_and_categories = []
        for index, paper in enumerate(papers, start=1):
            print(f"Processing paper {index}/{len(papers)}: {paper.title}")
            summary = summarize_abstract(paper.summary)
            rouge_scores = calculate_rouge(paper.summary, summary)
            print(f"Title: {paper.title}")
            print(f"Authors: {', '.join(author.name for author in paper.authors)}")
            print(f"Summary: {summary}")
            print(f"Link: {paper.entry_id}")
            category = categorize_abstract_with_llama(summary)
            print(f"Category: {category}")
            print(f"ROUGE Scores: {rouge_scores}")
            summaries_and_categories.append((summary, category, paper))

        categorized = {category: [] for category in [
            "Bayesian Statistics", "Computational Statistics", "Biostatistics",
            "Unsupervised Learning", "Supervised Learning",
            "High-Dimensional Statistics", "Time Series Analysis", "Multivariate Analysis",
            "Experimental Design", "Nonparametric Statistics", "Econometrics",
            "Probability Theory", "Statistical Learning Theory", "Applied Statistics",
            "Environmental Statistics", "Financial Statistics", "Survey Statistics",
            "Spatial Statistics", "Stochastic Processes", "Data Mining", "Neural Networks",
            "Reinforcement Learning", "Ensemble Learning", "Inferential Statistics",
            "Descriptive Statistics", "Machine Learning", "Statistics Sampling", "Bioinformatics",
            "Statistical Decision Theory", "Casual Inference", "Robust Statistics", "Uncategorized"
        ]}

        # Categorize each summary and paper
        for summary, category, paper in summaries_and_categories:
            if category not in categorized:
                categorized["Uncategorized"].append((summary, paper))
            else:
                categorized[category].append((summary, paper))
        
        print("All papers processed.")
        newsletter = create_newsletter(categorized, summaries_and_categories)
        print(newsletter)

    except Exception as e:
        print(f"An error occurred: {e}")

if __name__ == "__main__":
    main()
