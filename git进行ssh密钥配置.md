# git进行ssh密钥配置

这个错误提示表明，你在使用 SSH 连接 GitHub 时，Git 尝试验证 GitHub 的主机密钥，但系统无法确认其真实性，因此出现了 **Host key verification failed** 的问题。

### 解决步骤：

#### 1. **确认是否信任 GitHub 的主机密钥**
当你第一次使用 SSH 连接到 GitHub 时，系统会要求你确认 GitHub 服务器的主机密钥（这是为了确保你连接到的是合法的 GitHub 服务器，而不是中间人攻击）。为了继续，你需要手动确认并信任这个主机密钥。

当你看到以下提示：
```
The authenticity of host 'github.com (20.205.243.166)' can't be established.
ED25519 key fingerprint is SHA256:+DiY3wvvV6TuJJhbpZisF/zLDA0zPMSvHdkr4UvCOqU.
```

你可以输入 `yes` 来继续确认：
```
Are you sure you want to continue connecting (yes/no/[fingerprint])?
```
直接输入 `yes` 并按下回车键，将这个密钥添加到你的已知主机列表中。

#### 2. **检查 SSH 密钥是否正确配置**
如果你使用的是 SSH 连接，并且在推送时出现了 **Could not read from remote repository**，有可能是你的 SSH 密钥配置不正确。

- **检查是否已经配置 SSH 密钥**：
  ```bash
  ls ~/.ssh
  ```
  检查是否有类似 `id_rsa` 或 `id_ed25519` 的 SSH 密钥文件。如果没有，可以通过以下命令生成一个新的 SSH 密钥：

  ```bash
  ssh-keygen -t ed25519 -C "your_email@example.com"
  ```

- **将公钥添加到 GitHub**：
  使用以下命令显示生成的 SSH 公钥，并将其添加到 GitHub 账户的 SSH 设置中：
  ```bash
  cat ~/.ssh/id_ed25519.pub
  ```

  然后，将输出的公钥复制到 GitHub 网站上：
  - 登录 GitHub
  - 前往 "Settings" -> "SSH and GPG keys"
  - 点击 "New SSH key" 并将公钥粘贴到 GitHub 上。

#### 3. **清除旧的 GitHub 主机密钥**
如果你已经确认了主机密钥，仍然收到 **Host key verification failed** 错误，可能是 `~/.ssh/known_hosts` 文件中已经保存了旧的 GitHub 主机密钥。

你可以手动删除旧的条目，然后再次尝试连接。使用以下命令删除 GitHub 主机密钥：
```bash
ssh-keygen -R github.com
```

然后再次尝试 `git push`，这时系统会提示你确认新的 GitHub 主机密钥，输入 `yes` 即可。

#### 4. **检查远程仓库的访问权限**
确保你有正确的权限访问 GitHub 仓库。你可以通过以下命令检查当前配置的远程仓库 URL：
```bash
git remote -v
```

如果你看到的是 HTTPS URL，但希望使用 SSH，你可以通过以下命令将远程仓库 URL 更改为 SSH 格式：
```bash
git remote set-url origin git@github.com:Shirley-zzb/Notes.git
```

#### 5. **验证 SSH 连接**
使用以下命令验证你是否可以通过 SSH 连接到 GitHub：
```bash
ssh -T git@github.com
```

如果成功，你应该看到如下信息：
```
Hi [Your Username]! You've successfully authenticated, but GitHub does not provide shell access.
```

### 总结：
1. 当提示你确认 GitHub 主机密钥时，输入 `yes` 来继续。
2. 确保你的 SSH 密钥正确配置，并将公钥添加到 GitHub。
3. 如果出现主机密钥冲突，使用 `ssh-keygen -R github.com` 来删除旧的主机密钥。
4. 确保远程仓库 URL 正确，并验证你是否有权限访问该仓库。

完成以上步骤后，你应该可以顺利推送代码到 GitHub。如果仍然有问题，请告诉我具体的错误信息。

从你提供的 `~/.ssh` 目录内容来看，当前目录下没有 SSH 密钥对（如 `id_rsa`、`id_ed25519` 等），这意味着你可能还没有生成或配置 SSH 密钥。如果你打算使用 SSH 推送代码到 GitHub，需要先生成一个 SSH 密钥对并将公钥添加到 GitHub。

### 解决步骤：

#### 1. **生成 SSH 密钥对**
如果你还没有生成 SSH 密钥，可以使用以下命令生成新的 SSH 密钥对：

```bash
ssh-keygen -t ed25519 -C "your_email@example.com"
```

- `-t ed25519`：指定使用 `ed25519` 类型密钥（推荐）。
- `-C "your_email@example.com"`：注释部分，可以填上你的 GitHub 注册邮箱。

执行命令后，系统会提示你保存密钥的文件路径，按 Enter 键选择默认路径（`~/.ssh/id_ed25519`），并根据需要设置密码（如果不需要密码保护，直接按 Enter 跳过）。

#### 2. **将公钥添加到 GitHub**

生成 SSH 密钥对后，公钥会保存在 `~/.ssh/id_ed25519.pub` 文件中。你需要将此公钥添加到 GitHub 的 SSH 设置中。

使用以下命令显示公钥内容：

```bash
cat ~/.ssh/id_ed25519.pub
```

复制输出的公钥内容。

然后，前往 GitHub：
- 登录你的 GitHub 账户。
- 点击右上角的头像，选择 **Settings**。
- 在左侧菜单中选择 **SSH and GPG keys**。
- 点击 **New SSH key**，将刚才复制的公钥粘贴到公钥框中，并给它起个名字（如 "Office Computer"）。
- 点击 **Add SSH key**。

#### 3. **配置 SSH 连接**

如果你已经设置了 `~/.ssh/config` 文件，可以确保其中的配置正确，比如如下格式：

```bash
Host github.com
  HostName github.com
  User git
  IdentityFile ~/.ssh/id_ed25519
```

这个配置会确保你的 SSH 连接使用生成的 `id_ed25519` 密钥。

#### 4. **测试 SSH 连接到 GitHub**

完成配置后，你可以测试 SSH 连接到 GitHub：

```bash
ssh -T git@github.com
```

如果成功，你应该看到如下信息：
```
Hi [Your Username]! You've successfully authenticated, but GitHub does not provide shell access.
```

#### 5. **设置远程仓库为 SSH**

如果你当前的 Git 远程仓库使用的是 HTTPS 而不是 SSH，你可以通过以下命令将远程仓库 URL 切换为 SSH 格式：

```bash
git remote set-url origin git@github.com:Shirley-zzb/Notes.git
```

然后你可以尝试推送代码：

```bash
git push
```

### 总结：
1. **生成 SSH 密钥对**（如果尚未生成）。
2. **将公钥添加到 GitHub**。
3. **测试 SSH 连接**确保 GitHub 连接成功。
4. **使用 SSH URL 进行代码推送**。

如果你按照这些步骤进行操作，应该能够解决连接到 GitHub 的问题。如果有任何进一步的问题，请告诉我！