# Dans report_planner.py
import config
from data_models import ReportPlan, ReportSection
from typing import List, Optional
import logging
import uuid # <-- Importer UUID

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - [%(module)s] - %(message)s')
log = logging.getLogger(__name__)

class ReportPlanner:
    def __init__(self):
        self.default_structure = config.DEFAULT_REPORT_STRUCTURE

    def create_base_plan(self, structure_definition: Optional[List[str]] = None) -> ReportPlan:
        if structure_definition is None:
            structure_definition = self.default_structure
            log.info("Using default report structure from config.")

        report_sections = []
        section_stack = [] # Pour gérer la hiérarchie

        for title_raw in structure_definition:
            title = title_raw.strip()
            level = 1 # Niveau par défaut
            if title_raw.startswith("   "): level = 3; title = title.lstrip()
            elif title_raw.startswith(" "): level = 2; title = title.lstrip()

            # !! AJOUT : Générer un ID unique !!
            section_id = f"sec_{uuid.uuid4().hex[:8]}" # Exemple: sec_a1b2c3d4
            # !! FIN AJOUT !!

            # Créer la section avec l'ID
            section = ReportSection(section_id=section_id, title=title, level=level, status="pending", subsections=[]) # Assurer que subsections est une liste vide

            # Gérer la hiérarchie basée sur le niveau
            while section_stack and section_stack[-1].level >= level:
                section_stack.pop() # Remonter dans la hiérarchie

            if not section_stack:
                # Niveau racine
                report_sections.append(section)
            else:
                # Ajouter comme sous-section du dernier élément de la pile
                parent_section = section_stack[-1]
                # Assurer que subsections est une liste avant d'ajouter
                if not hasattr(parent_section, 'subsections') or parent_section.subsections is None:
                     parent_section.subsections = []
                parent_section.subsections.append(section)

            # Ajouter la section actuelle à la pile
            section_stack.append(section)

        plan = ReportPlan(title="Apprenticeship Report", structure=report_sections)
        log.info("Generated basic report plan with unique section IDs.")
        return plan