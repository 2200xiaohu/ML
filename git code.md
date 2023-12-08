# 第一次上传

```
cd

git init

#如果要创建branch
git branch my-branch

# switch to that branch (line of development)
git checkout my-branch


git add .

git config --global user.email "295464699@qq.com"

git commit -m "init"

git push --set-upstream origin my-branch
```



# 持续更新

```
cd

git status

git add -A
git commit -a -m "update"

git push origin my-branch -f
```

