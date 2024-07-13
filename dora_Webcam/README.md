# 使用文档

## cargo

``` C++
// 查看cargo安装的库
cargo install --list
```

## dora

``` python
# 启动 coordinator daemon
dora up

# 关闭
dora stop

# 启动数据流
# --attach：等待数据流完成后再返回
# --hot-reload：在数据流运行时修改Python Operator
# --name：命名数据流
dora start dataflow.yml --attach --hot-reload --name webcam

# 关闭数据流
dora stop --name webcam

# 查看节点输出
dora logs webcam webcam
```

``` C++
$ dora --help
dora-rs cli client

Usage: dora.exe <COMMAND>

Commands:
  check        Check if the coordinator and the daemon is running
  graph        Generate a visualization of the given graph using mermaid.js. Use --open to open browser
  build        Run build commands provided in the given dataflow
  new          Generate a new project, node or operator. Choose the language between Rust, Python, C or C++
  up           Spawn a coordinator and a daemon
  destroy      Destroy running coordinator and daemon. If some dataflows are still running, they will be stopped first
  start        Start the given dataflow path. Attach a name to the running dataflow by using --name
  stop         Stop the given dataflow UUID. If no id is provided, you will be able to choose between the running dataflows
  list         List running dataflows
  logs         Show logs of a given dataflow and node
  daemon       Run daemon
  runtime      Run runtime
  coordinator  Run coordinator
  help         Print this message or the help of the given subcommand(s)

Options:
  -h, --help     Print help
  -V, --version  Print version
```
