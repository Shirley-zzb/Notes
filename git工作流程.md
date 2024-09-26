# git工作流程

### Git 工作流程

1. **初始化仓库**
   ```bash
   git init
   ```

2. **克隆远程仓库**
   ```bash
   git clone <repository-url>
   ```

3. **创建新分支**
   ```bash
   git checkout -b <branch-name>
   ```

4. **查看分支**
   ```bash
   git branch
   ```

5. **添加更改**
   ```bash
   git add <file>  # 添加特定文件
   git add .       # 添加所有更改的文件
   ```

6. **提交更改**
   ```bash
   git commit -m "Your commit message"
   ```

7. **查看提交历史**
   ```bash
   git log
   ```

8. **合并分支**
   - 切换到要合并到的分支：
     ```bash
     git checkout <target-branch>
     ```
   - 合并：
     ```bash
     git merge <branch-name>
     ```

9. **推送到远程仓库**
   ```bash
   git push origin <branch-name>
   ```

10. **拉取远程更改**
    ```bash
    git pull origin <branch-name>
    ```

11. **删除分支**
    ```bash
    git branch -d <branch-name>  # 本地删除
    git push origin --delete <branch-name>  # 远程删除
    ```

### 本地 Git 仓库工作流程

1. **初始化本地仓库**
   
   - 创建一个新目录并进入：
     ```bash
     mkdir <repository-name>
     cd <repository-name>
     ```
   - 初始化 Git 仓库：
     ```bash
     git init
     ```
   
2. **创建文件并添加内容**
   - 创建一个文件（例如 README.md）并添加一些内容。

3. **查看当前状态**
   ```bash
   git status
   ```

4. **添加更改**
   - 将文件添加到暂存区：
     ```bash
     git add README.md  # 添加特定文件
     git add .         # 添加所有更改的文件
     ```

5. **提交更改**
   ```bash
   git commit -m "Initial commit"
   ```

6. **创建新分支**
   ```bash
   git checkout -b <branch-name>
   ```

7. **进行修改并提交**
   - 修改文件，然后再次添加和提交：
     ```bash
     git add <modified-file>
     git commit -m "Description of changes"
     ```

8. **查看提交历史**
   ```bash
   git log
   ```

9. **合并分支**
   - 切换回主分支（通常是 `main` 或 `master`）：
     ```bash
     git checkout main
     ```
   - 合并你的新分支：
     ```bash
     git merge <branch-name>
     ```

10. **删除分支**
    ```bash
    git branch -d <branch-name>  # 删除本地分支
    ```

### 常用命令总结

- **查看状态**: `git status`
- **查看历史**: `git log`
- **切换分支**: `git checkout <branch-name>`
- **添加文件**: `git add <file>` 或 `git add .`
- **提交更改**: `git commit -m "message"`
- **查看分支**: `git branch
