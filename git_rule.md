# GymExcerciseCorrection

## Technology


## Clone Project


## Git Flow

- Step 1: Update status of task in [trello]
- Step 2: checkout to `main`, pull newest code from `main`
    ```
    git checkout main
    git pull origin main
    ```
- Step 3: Create branch for task, base in branch `main`

    **Rule of branch name:**

    - If `Tracker` in trello is `'Feature'` or `'Subtask'` of Feature, branch name start with `feat/`
    - If `Tracker` in trello is `'Bug'` or `'Subtask'` of Bug, branch name start with `fix/`
    - If other, branch name start with `task/`

    Example: Issue in trello have Tracker is `'Subtask'` of Feature, Name is `BE Create Login API`. Branch name is `feat/pbl5-be-create-login-api`
    ```
    git checkout -b feat/pbl5-be-create-login-api main
    ```
- Step 4: When commit, message of commit follow rule
    - If column `Tracker` in Redmine is `'Feature'`, branch name start with `feat: `
    - If column `Tracker` in Redmine is `'Bug'`, branch name start with `fix: `
    - If other, branch name start with `task: `
    - Next is string `[PBL5]`
    - Next is commit content

    Example: `feat: [PBL5] Coding page login` or `fix: [PBL5] Coding page login`
- Step 5: When create merge request
    
    **Rule of merge request name:**
    
    - Start with `[PBL5]`
    - Next is  merge request content

        Example: `[PBL5] Coding page login`

    **Rule of merge request description:**

    - In **`What does this MR do and why?`**, replace _`Describe in detail what your merge request does and why.`_ with your content of this merge request
    - In **`Screenshots or screen recordings`**, replace _`These are strongly recommended to assist reviewers and reduce the time to merge your change.`_ with screen recordings of feature or task for this merge request
    - Check the checklist
    - Select approver
    - Select merger
- Note:
    - If have conflict with branch `develop` or need to get newest code from branch `develop`
    - Fix conflict and create merge request from `pbl5/dev` to `develop`
