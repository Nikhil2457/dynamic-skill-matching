import streamlit as st
import pandas as pd
import os
import nltk
nltk.data.path.append("./nltk_data")

from matching import match_skills


def load_data():
    """Load data with comprehensive error handling"""
    try:
        script_dir = os.path.dirname(os.path.abspath(__file__))
        users_path = os.path.join(script_dir, "users.csv")
        projects_path = os.path.join(script_dir, "projects.csv")

        if not os.path.exists(users_path):
            raise FileNotFoundError(f"users.csv not found at {users_path}")
        if not os.path.exists(projects_path):
            raise FileNotFoundError(f"projects.csv not found at {projects_path}")

        users = pd.read_csv(users_path)
        projects = pd.read_csv(projects_path)

        required_user_cols = {'user_id', 'skills', 'experience'}
        required_project_cols = {'project_id', 'requirements'}

        if not required_user_cols.issubset(users.columns):
            missing = required_user_cols - set(users.columns)
            raise ValueError(f"users.csv missing columns: {missing}")
        if not required_project_cols.issubset(projects.columns):
            missing = required_project_cols - set(projects.columns)
            raise ValueError(f"projects.csv missing columns: {missing}")

        return users, projects

    except Exception as e:
        st.error(f"Data loading failed: {str(e)}")
        st.stop()


def main():
    st.set_page_config(page_title="AI-Powered Team Builder", layout="wide")
    st.title("🤖 AI-Powered Dynamic Team Builder")

    # Load users and projects
    users, projects = load_data()

    # Extract all unique skills from project requirements
    try:
        all_skills = sorted(set(
            skill.strip().title()
            for req in projects['requirements'].astype(str)
            for skill in req.replace(',', ' and ').split(' and ')
            if skill.strip()
        ))
    except Exception as e:
        st.error(f"Failed to extract skills: {str(e)}")
        st.stop()

    # UI - Skill and team size selection
    st.markdown("### Step 1: Select Skills for Your Project")
    selected_skills = st.multiselect("Choose skills needed", all_skills)

    skill_requirements = {}
    if selected_skills:
        st.markdown("### Step 2: Specify Number of People Required for Each Skill")
        for skill in selected_skills:
            count = st.number_input(f"👥 {skill}", min_value=1, max_value=10, value=1, key=f"{skill}_count")
            skill_requirements[skill] = count

    # Button to trigger team building
    if st.button("🚀 Build Team"):
        with st.spinner("Finding the best team..."):
            final_team = pd.DataFrame()
            used_user_ids = set()
            summary_rows = []
            index = 1  # Start from 1 for serial numbers

            for skill, num_needed in skill_requirements.items():
                dummy_project = pd.Series({
                    'project_id': 0,
                    'requirements': skill
                })

                matches = match_skills(users, dummy_project)

                # Exclude already selected users
                matches = matches[~matches['user_id'].isin(used_user_ids)]

                if matches.empty:
                    summary_rows.append({
                        'Serial No': index,  # Add serial number starting from 1
                        'Skill': skill,
                        'Needed': num_needed,
                        'Assigned': 0,
                        'Status': "❌ No one available"
                    })
                    index += 1  # Increment serial number
                    continue

                # Sort by experience (level/rate) in descending order
                matches_sorted = matches.sort_values(by='experience', ascending=False)

                # Select the top `num_needed` matches
                top_matches = matches_sorted.head(num_needed)
                used_user_ids.update(top_matches['user_id'])

                top_matches = top_matches.copy()
                final_team = pd.concat([final_team, top_matches], ignore_index=True)

                summary_rows.append({
                    'Serial No': index,  # Add serial number starting from 1
                    'Skill': skill,
                    'Needed': num_needed,
                    'Assigned': len(top_matches),
                    'Status': "✅ Filled" if len(top_matches) == num_needed else "⚠️ Partially Filled"
                })
                index += 1  # Increment serial number

            # Summary Table
            st.markdown("### 📊 Assignment Summary")
            st.dataframe(pd.DataFrame(summary_rows), use_container_width=True)

            # Modify final team table to add Serial No.
            if final_team.empty:
                st.warning("⚠️ No team members matched the selected skills.")
            else:
                st.success("✅ Final team formed!")
                st.markdown("### 🧠 Final Team Members")

                # Modify the columns for final team
                final_team['Serial No'] = range(1, len(final_team) + 1)
                final_team = final_team[['Serial No', 'skills', 'experience', 'match_score']]
                
                # Rename 'experience' to 'Level/Rate'
                final_team = final_team.rename(columns={'experience': 'Level/Rate'})

                st.dataframe(
                    final_team[['Serial No', 'skills', 'Level/Rate', 'match_score']],
                    column_config={
                        "match_score": st.column_config.ProgressColumn(
                            "Match Score",
                            help="Strength of skill match (0-1)",
                            format="%.2f",
                            min_value=0,
                            max_value=1,
                        )
                    },
                    hide_index=True,
                    use_container_width=True
                )

                # Download CSV
                csv = final_team.to_csv(index=False).encode('utf-8')
                st.download_button("⬇️ Download Team as CSV", data=csv, file_name="final_team.csv", mime='text/csv')


if __name__ == "__main__":
    main()
