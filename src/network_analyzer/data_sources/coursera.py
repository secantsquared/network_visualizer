"""
Coursera course data source adapter.
"""

import json
import pandas as pd
from pathlib import Path
from typing import List, Dict
from .base import DataSourceAdapter


class CourseraDataSource(DataSourceAdapter):
    """Coursera course data source adapter using Kaggle dataset."""
    
    def __init__(self, config, dataset_path: str):
        super().__init__(config)
        self.dataset_path = Path(dataset_path)
        self.coursera_df = None
        self.skill_to_courses = {}
        self.course_to_skills = {}
        self.course_to_metadata = {}
        
        # Load and process dataset
        self._load_dataset()
        self._build_mappings()
    
    def _load_dataset(self):
        """Load Coursera dataset from CSV."""
        try:
            self.coursera_df = pd.read_csv(self.dataset_path)
            self.logger.info(f"Loaded {len(self.coursera_df)} courses from {self.dataset_path}")
        except Exception as e:
            self.logger.error(f"Failed to load dataset from {self.dataset_path}: {e}")
            raise
    
    def _build_mappings(self):
        """Build skill-course mappings from dataset."""
        for idx, row in self.coursera_df.iterrows():
            course_title = row['Title']
            skills = self._parse_skills(row.get('Skills', ''))
            
            # Course to skills mapping
            self.course_to_skills[course_title] = skills
            
            # Skill to courses mapping
            for skill in skills:
                if skill not in self.skill_to_courses:
                    self.skill_to_courses[skill] = []
                self.skill_to_courses[skill].append(course_title)
            
            # Course metadata
            self.course_to_metadata[course_title] = {
                'organization': row.get('Organization', ''),
                'rating': row.get('Ratings', 0),
                'review_count': row.get('Review count', 0),
                'difficulty': self._extract_difficulty(row.get('Miscellaneous info', '')),
                'duration': self._extract_duration(row.get('Miscellaneous info', '')),
                'type': self._extract_type(row.get('Miscellaneous info', '')),
                'skills': skills
            }
    
    def _parse_skills(self, skills_string) -> List[str]:
        """Parse skills from dataset format."""
        if pd.isna(skills_string) or not skills_string:
            return []
        
        try:
            # Handle different formats in the dataset
            if skills_string.startswith('['):
                # JSON-like format
                return json.loads(skills_string.replace("'", '"'))
            else:
                # Comma-separated format
                return [s.strip() for s in str(skills_string).split(',') if s.strip()]
        except Exception as e:
            self.logger.warning(f"Failed to parse skills: {skills_string}. Error: {e}")
            return []
    
    def _extract_difficulty(self, misc_info: str) -> str:
        """Extract difficulty level from miscellaneous info."""
        misc_lower = str(misc_info).lower()
        if 'beginner' in misc_lower:
            return 'Beginner'
        elif 'intermediate' in misc_lower:
            return 'Intermediate'
        elif 'advanced' in misc_lower:
            return 'Advanced'
        return 'Unknown'
    
    def _extract_duration(self, misc_info: str) -> str:
        """Extract duration from miscellaneous info."""
        # Look for patterns like "4 weeks", "20 hours", etc.
        import re
        duration_pattern = r'(\d+)\s*(week|month|hour|day)s?'
        match = re.search(duration_pattern, str(misc_info).lower())
        if match:
            return f"{match.group(1)} {match.group(2)}s"
        return 'Unknown'
    
    def _extract_type(self, misc_info: str) -> str:
        """Extract course type from miscellaneous info."""
        misc_lower = str(misc_info).lower()
        if 'specialization' in misc_lower:
            return 'Specialization'
        elif 'course' in misc_lower:
            return 'Course'
        elif 'certificate' in misc_lower:
            return 'Certificate'
        return 'Course'
    
    def get_relationships(self, item: str) -> List[str]:
        """Get related courses based on shared skills. Item can be a course title or skill name."""
        related_courses = set()
        self.logger.debug(f"Finding relationships for item: '{item}'")
        
        # Check if item is a course title
        if item in self.course_to_skills:
            # Item is a course title - find related courses by shared skills
            course_skills = self.course_to_skills[item]
            self.logger.debug(f"'{item}' is a course with skills: {course_skills}")
            for skill in course_skills:
                skill_courses = self.skill_to_courses.get(skill, [])
                related_courses.update(skill_courses)
                self.logger.debug(f"Skill '{skill}' has {len(skill_courses)} courses")
            # Remove self
            related_courses.discard(item)
        
        # Check if item is a skill name
        elif item in self.skill_to_courses:
            # Item is a skill name - return courses that teach this skill
            courses_teaching_skill = self.skill_to_courses[item]
            self.logger.debug(f"'{item}' is a skill taught by {len(courses_teaching_skill)} courses")
            related_courses.update(courses_teaching_skill)
            
            # Also find courses that teach related skills
            for course in courses_teaching_skill[:3]:  # Limit to avoid too many relationships
                if course in self.course_to_skills:
                    course_skills = self.course_to_skills[course]
                    for skill in course_skills:
                        if skill != item:  # Don't include the original skill
                            skill_courses = self.skill_to_courses.get(skill, [])
                            related_courses.update(skill_courses)
        
        # If no direct matches, try partial matching for skills
        if not related_courses:
            item_lower = item.lower()
            partial_matches = []
            for skill in self.skill_to_courses:
                skill_lower = skill.lower()
                if item_lower in skill_lower or skill_lower in item_lower:
                    partial_matches.append((skill, len(self.skill_to_courses[skill])))
            
            # Sort by number of courses (prefer more popular skills)
            partial_matches.sort(key=lambda x: x[1], reverse=True)
            
            # Take the best match
            if partial_matches:
                best_skill = partial_matches[0][0]
                related_courses.update(self.skill_to_courses[best_skill])
                self.logger.info(f"Used partial match '{best_skill}' for '{item}'")
        
        # If still no matches, try fuzzy matching on course titles
        if not related_courses:
            item_lower = item.lower()
            for course_title in self.course_to_skills:
                if item_lower in course_title.lower() or course_title.lower() in item_lower:
                    related_courses.add(course_title)
                    # Add courses with similar skills to this one
                    if course_title in self.course_to_skills:
                        course_skills = self.course_to_skills[course_title]
                        for skill in course_skills[:2]:  # Limit to avoid explosion
                            related_courses.update(self.skill_to_courses.get(skill, []))
                    break
        
        # Convert to list and limit
        related_list = list(related_courses)
        final_list = related_list[:self.config.links_per_article]
        self.logger.debug(f"Returning {len(final_list)} relationships for '{item}': {final_list[:3]}...")
        return final_list
    
    def get_skill_prerequisites(self, skill: str) -> List[str]:
        """Get prerequisite skills based on course difficulty progression."""
        prerequisites = []
        
        # Find courses teaching this skill
        teaching_courses = self.skill_to_courses.get(skill, [])
        
        # Look for beginner-level courses teaching foundational skills
        for course in teaching_courses:
            course_meta = self.course_to_metadata.get(course, {})
            if course_meta.get('difficulty') == 'Beginner':
                # This is a beginner course, its skills might be prerequisites
                course_skills = self.course_to_skills.get(course, [])
                prerequisites.extend(course_skills)
        
        return list(set(prerequisites))
    
    def should_filter_item(self, course_title: str) -> bool:
        """Check if course should be filtered out."""
        if not course_title:
            return True
        
        # Check title length
        if (len(course_title) < self.config.min_title_length or 
            len(course_title) > self.config.max_title_length):
            return True
        
        # Check for unwanted patterns
        title_lower = course_title.lower()
        for pattern in self.config.filter_patterns:
            if pattern.lower() in title_lower:
                return True
        
        # Filter courses with very low ratings (if available)
        course_meta = self.course_to_metadata.get(course_title, {})
        rating = course_meta.get('rating', 0)
        if isinstance(rating, (int, float)) and rating < 3.0 and rating > 0:
            return True
        
        return False
    
    def get_item_metadata(self, course_title: str) -> Dict:
        """Get metadata for course."""
        base_metadata = {
            'type': 'coursera_course',
            'title': course_title,
            'source': 'Coursera'
        }
        
        course_meta = self.course_to_metadata.get(course_title, {})
        base_metadata.update(course_meta)
        
        return base_metadata
    
    def get_source_type(self) -> str:
        return "coursera"
    
    def get_courses_by_skill(self, skill: str) -> List[str]:
        """Get courses that teach a specific skill."""
        return self.skill_to_courses.get(skill, [])
    
    def get_courses_by_difficulty(self, difficulty: str) -> List[str]:
        """Get courses filtered by difficulty level."""
        courses = []
        for course, metadata in self.course_to_metadata.items():
            if metadata.get('difficulty') == difficulty:
                courses.append(course)
        return courses
    
    def get_learning_path_for_skills(self, target_skills: List[str], 
                                   current_level: str = "Beginner") -> List[str]:
        """Get a learning path for acquiring specific skills."""
        path_courses = []
        
        for skill in target_skills:
            # Get courses teaching this skill
            skill_courses = self.get_courses_by_skill(skill)
            
            # Filter by appropriate difficulty level
            for course in skill_courses:
                course_meta = self.course_to_metadata.get(course, {})
                if course_meta.get('difficulty') == current_level:
                    path_courses.append(course)
        
        return list(set(path_courses))