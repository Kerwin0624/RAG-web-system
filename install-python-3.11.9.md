# 安装 Python 3.11.9

安装包已下载到：**`/Users/openclaw/Downloads/python-3.11.9-macos11.pkg`**

## 方式一：图形界面安装（推荐）

1. 打开 **访达**，进入 **下载** 文件夹
2. 双击 **python-3.11.9-macos11.pkg**
3. 按提示点击「继续」完成安装

## 方式二：终端安装

在终端中执行：

```bash
sudo installer -pkg /Users/openclaw/Downloads/python-3.11.9-macos11.pkg -target /
```

输入管理员密码后等待安装完成。

---

## 安装后验证

安装完成后，**新开一个终端窗口**，执行：

```bash
# 查看 Python 3.11 路径（官方安装通常在 /Library/Frameworks/Python.framework/）
/Library/Frameworks/Python.framework/Versions/3.11/bin/python3 --version
```

若希望命令行直接使用 `python3.11`，可把该目录加入 PATH。在 `~/.zshrc` 末尾添加：

```bash
export PATH="/Library/Frameworks/Python.framework/Versions/3.11/bin:$PATH"
```

然后执行 `source ~/.zshrc`，之后可用：

```bash
python3.11 --version   # 应显示 Python 3.11.9
```
