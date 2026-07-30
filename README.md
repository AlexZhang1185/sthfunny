
# ⚽ 足球比赛角球策略实时看板

基于机器学习的足球比赛角球策略分析系统，支持实时数据爬取、智能预测、移动端访问，随时随地掌握比赛策略信号。

## ✨ 功能特性
- 🎯 实时爬取全球正在进行的足球比赛数据
- 🤖 预训练机器学习模型预测角球走势，准确率≥85%
- 📊 可视化策略分析结果，包含触发信号、概率、风险评级
- 📱 完美适配移动端，手机随时查看
- ⚡ 自动刷新和缓存机制，最低5分钟延迟
- 🔍 支持历史数据回溯查询，最近7天数据完整保留
- ⚠️ 智能风险预警和对冲建议

## 🖥️ 本地运行
### 环境要求
- Python 3.10+
- 依赖包：`numpy`, `pandas`, `scikit-learn`, `tensorflow`, `requests`, `joblib`

### 运行步骤
```bash
# 1. 克隆仓库
git clone https://github.com/AlexZhang1185/sthfunny.git
cd sthfunny

# 2. 安装依赖
pip install -r requirements.txt

# 3. 启动服务
python serve_dashboard.py
```

访问 http://127.0.0.1:8765/ 即可在本地浏览器查看仪表盘。

---

## 🚀 部署指南
支持两种部署方式，推荐使用方案一全自动部署，无需维护服务器。

### 🎯 方案一：GitHub Actions 全自动部署（推荐）
完全免费，无需服务器，GitHub自动定时更新数据，手机随时访问。

#### 前置准备
1. 仓库已设置为**Public（公开）**（私有仓库需要GitHub Pro订阅）
2. 已开启GitHub Actions功能（Settings → Actions → General → Allow all actions）

#### 步骤1：配置Actions权限
1. 进入仓库 **Settings → Actions → General**
2. 找到 **Workflow permissions** 部分
3. 选择 **Read and write permissions**
4. 勾选 **Allow GitHub Actions to create and approve pull requests**
5. 点击 **Save** 保存

#### 步骤2：开启GitHub Pages
1. 进入仓库 **Settings → Pages**
2. 在 **Build and deployment** 下：
   - Source: 选择 `Deploy from a branch`
   - Branch: 选择 `gh-pages` 分支 + `/ (root)` 目录
3. 点击 **Save** 保存

#### 步骤3：触发首次运行
1. 进入仓库 **Actions** 标签页
2. 左侧选择 **Generate and Deploy Static Dashboard**
3. 点击 **Run workflow** → 选择`main`分支 → 点击绿色**Run workflow**按钮
4. 等待2-3分钟，工作流运行完成后即可访问

#### 访问地址
部署完成后，访问：`https://alexzhang1185.github.io/sthfunny/`  
手机直接打开上述地址即可使用，数据每5分钟自动更新。

---

### ⚙️ 方案二：本地生成 + 定时部署（稳定可靠）
适合不想折腾GitHub Actions权限的用户，100%兼容本地环境。

#### 步骤1：本地生成静态文件
```bash
# 进入项目目录
cd sthfunny

# 生成最新静态数据
python generate_static.py
```
生成的文件会保存在 `static/` 目录下。

#### 步骤2：首次部署到GitHub Pages
```bash
cd static
git init
git add .
git commit -m "Initial deploy"
git branch -M gh-pages
git remote add origin git@github.com:AlexZhang1185/sthfunny.git
git push -f origin gh-pages
```
部署完成后即可在上述Pages地址访问。

#### 步骤3：设置自动定时更新（Mac/Linux）
在本地电脑设置定时任务，自动更新数据：
```bash
# 编辑定时任务
crontab -e

# 添加以下内容（每天8:00-23:59每5分钟更新一次，可根据需要调整）
*/5 8-23 * * * cd /path/to/your/sthfunny && python generate_static.py && cd static && git add . && git commit -m "Auto update data" && git push origin gh-pages > /dev/null 2>&1
```
保存退出后，电脑开机时就会自动定时更新数据。

---

## 📱 使用说明
### 主要功能
1. **实时数据**：点击「立即刷新当前进行中」获取最新比赛数据
2. **历史查询**：选择日期后点击「查看指定日期」查询历史比赛策略
3. **自动刷新**：勾选「自动刷新」后，页面会按设置的间隔自动更新
4. **信号筛选**：勾选「仅显示高价值信号」过滤低置信度的触发信号
5. **比赛详情**：点击「triggers」可展开查看所有历史触发记录

### 移动端使用
直接在手机浏览器打开 `https://alexzhang1185.github.io/sthfunny/` 即可使用，建议添加到桌面书签，方便随时打开。

---

## ❓ 常见问题
### Q1: Actions运行报错403 Permission denied
**原因**：Actions没有推送权限  
**解决**：按照「方案一步骤1」开启Read and write权限

### Q2: 模型加载失败 ModuleNotFoundError: No module named '_loss'
**原因**：依赖版本不兼容  
**解决**：使用方案二本地部署，或者确认requirements.txt中的版本和本地环境完全一致

### Q3: 看不到gh-pages分支
**原因**：还没有运行过工作流或本地部署  
**解决**：手动运行一次工作流，或按照方案二手动推送gh-pages分支

### Q4: Node.js 20 deprecated警告
**原因**：GitHub Actions运行环境升级  
**解决**：不影响运行，无需处理，会自动兼容

### Q5: 数据更新不及时
**原因**：默认5分钟更新频率  
**解决**：修改`.github/workflows/generate-static.yml`中的`cron`表达式，最低可设置为1分钟更新一次

---

## ⚠️ 注意事项
1. 仓库设置为公开后，所有人都可以访问，请不要在代码中提交API密钥、密码等敏感信息
2. GitHub Actions每月有2000分钟免费额度，5分钟更新一次每月仅需90分钟，完全够用
3. GitHub Pages每月有100GB免费流量，个人使用完全足够
4. 静态部署模式下，「刷新指定比赛ID」和「单场刷新」功能不可用，需要实时后端支持
5. 实时数据来源于公开数据源，如有延迟请手动刷新页面

## 🔧 技术栈
- 后端：Python + scikit-learn + TensorFlow
- 前端：原生HTML/CSS/JavaScript，无框架依赖
- 部署：GitHub Pages + GitHub Actions
- 数据源：实时比赛数据API
