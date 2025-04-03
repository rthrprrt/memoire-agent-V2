import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import config
from data_models import JournalEntry
from typing import List, Dict
import datetime
import logging
import os

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

class Visualizer:
    """Creates visualizations based on report data."""

    def __init__(self):
        self.output_dir = config.OUTPUT_DIR
        os.makedirs(self.output_dir, exist_ok=True)

    def plot_competency_timeline(self, competency_timeline: Dict[str, List[datetime.date]], filename: str = "competency_timeline.png"):
        """Creates a timeline plot showing when competencies were mentioned."""
        logging.info("Generating competency timeline plot...")
        if not competency_timeline:
            logging.warning("No competency timeline data to plot.")
            return

        competencies = list(competency_timeline.keys())
        dates_by_competency = list(competency_timeline.values())

        # Filter out competencies with no dates
        valid_indices = [i for i, dates in enumerate(dates_by_competency) if dates]
        if not valid_indices:
             logging.warning("No competencies with associated dates found.")
             return

        competencies = [competencies[i] for i in valid_indices]
        dates_by_competency = [dates_by_competency[i] for i in valid_indices]


        # Determine the overall date range
        all_dates = [date for sublist in dates_by_competency for date in sublist]
        if not all_dates:
            logging.warning("No dates found in competency data.")
            return
        min_date = min(all_dates) - datetime.timedelta(days=7) # Add padding
        max_date = max(all_dates) + datetime.timedelta(days=7) # Add padding

        plt.style.use('seaborn-v0_8-gird') # Use an available style
        fig, ax = plt.subplots(figsize=(12, max(6, len(competencies) * 0.5))) # Adjust height based on number of competencies

        y_ticks = range(len(competencies))
        ax.set_yticks(y_ticks)
        ax.set_yticklabels(competencies)

        for i, dates in enumerate(dates_by_competency):
            # Convert datetime.date to numerical format for plotting
            num_dates = mdates.date2num(dates)
            ax.plot(num_dates, [i] * len(dates), 'o', markersize=5, label=competencies[i] if i < 10 else "") # Plot points

        # Formatting the x-axis (dates)
        ax.xaxis.set_major_locator(mdates.MonthLocator()) # Ticks for each month
        ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d')) # Format dates
        ax.set_xlim(mdates.date2num(min_date), mdates.date2num(max_date)) # Set date range
        plt.xticks(rotation=45)

        ax.set_xlabel("Date")
        ax.set_ylabel("Competency")
        ax.set_title("Competency Development Timeline")
        ax.grid(True, axis='x', linestyle='--', alpha=0.6)
        plt.tight_layout() # Adjust layout

        # Invert y-axis so top competency is at the top
        ax.invert_yaxis()

        output_path = os.path.join(self.output_dir, filename)
        try:
            plt.savefig(output_path)
            logging.info(f"Competency timeline plot saved to {output_path}")
        except Exception as e:
            logging.error(f"Error saving plot to {output_path}: {e}")
        plt.close(fig) # Close the plot figure


    def plot_project_activity(self, project_mentions: Dict[str, List[datetime.date]], filename: str = "project_activity.png"):
         """Creates a timeline showing mentions of different projects."""
         logging.info("Generating project activity plot...")
         # Similar structure to competency timeline
         if not project_mentions:
             logging.warning("No project mention data to plot.")
             return

         projects = list(project_mentions.keys())
         dates_by_project = list(project_mentions.values())

         valid_indices = [i for i, dates in enumerate(dates_by_project) if dates]
         if not valid_indices:
              logging.warning("No projects with associated dates found.")
              return

         projects = [projects[i] for i in valid_indices]
         dates_by_project = [dates_by_project[i] for i in valid_indices]

         all_dates = [date for sublist in dates_by_project for date in sublist]
         if not all_dates:
             logging.warning("No dates found in project data.")
             return
         min_date = min(all_dates) - datetime.timedelta(days=7)
         max_date = max(all_dates) + datetime.timedelta(days=7)

         plt.style.use('seaborn-v0_8-gird')
         fig, ax = plt.subplots(figsize=(12, max(6, len(projects) * 0.5)))

         y_ticks = range(len(projects))
         ax.set_yticks(y_ticks)
         ax.set_yticklabels(projects)

         for i, dates in enumerate(dates_by_project):
             num_dates = mdates.date2num(dates)
             ax.plot(num_dates, [i] * len(dates), 's', markersize=5, label=projects[i] if i < 10 else "") # Use squares 's'

         ax.xaxis.set_major_locator(mdates.MonthLocator())
         ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d'))
         ax.set_xlim(mdates.date2num(min_date), mdates.date2num(max_date))
         plt.xticks(rotation=45)

         ax.set_xlabel("Date")
         ax.set_ylabel("Project")
         ax.set_title("Project Mention Timeline")
         ax.grid(True, axis='x', linestyle='--', alpha=0.6)
         plt.tight_layout()
         ax.invert_yaxis()

         output_path = os.path.join(self.output_dir, filename)
         try:
             plt.savefig(output_path)
             logging.info(f"Project activity plot saved to {output_path}")
         except Exception as e:
             logging.error(f"Error saving plot to {output_path}: {e}")
         plt.close(fig)

    # Add more visualization methods as needed (e.g., tag frequency bar chart)