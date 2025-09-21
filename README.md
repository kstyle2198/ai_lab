# LangGraph Memos


## 1. Start LangGraph with State

LangGraph에서 가장 먼저 할 일은 Graph의 State를 정의하는 일입니다.<br>
State는 Graph에서 다룰 Data에 대한 Schema와 이 Data들에 대한 업데이트 적용 방법을 지정하는 함수(Reducer)로 구성됩니다.

Schema는 TypeDict, dataclass, Pydantic BaseModel 등을 활용해 정의할 수 있습니다. (각각의 특징은 패스) <br>
기본적으로 Graph는 동일한 인풋, 아웃풋 Schema를 갖습니다. <br>
개발자는 input용 schema와 output용 schema를 나누어 정의할 수 있습니다.

통상 모든 Graph node는 하나의 Schema를 사용합니다. <br>
이말은 모든 node가 동일한 State channel에 읽고, 쓴다는 의미입니다. <br>
단, 개발자는 특정 노드에서만 활용할 private schema를 정의하여 사용할 수 있습니다. <br>

이부분은 약간 이상하다 생각될 수 있습니다.<br>
통상 Graph를 build할 때, 아래와 같이 전역 State를 정의합니다.

```
builder = StateGraph(OverallState,input_schema=InputState,output_schema=OutputState)
```
여기에는 OverallState, InputState, OutputState의 세개의 Schema가 특정되었습니다.<br>
그런데, 여기에 포함되지 않았더라도, 별도로 정의된 Private Schema가 있다면,<br>
Graph안의 특정 노드는 이 Private Schema를 사용할 수 있습니다.<br>
그리고 전체 Graph 구조 안에서 Private와 OverallState간에 인풋, 아웃풋을 소통하면서 데이터를 업데이트 할 수 있습니다.

```
class InputState(TypedDict):
    user_input: str

class OutputState(TypedDict):
    graph_output: str

class OverallState(TypedDict):
    foo: str
    user_input: str
    graph_output: str

class PrivateState(TypedDict):
    bar: str

def node_1(state: InputState) -> OverallState:
    # Write to OverallState
    return {"foo": state["user_input"] + " name"}

def node_2(state: OverallState) -> PrivateState:
    # Read from OverallState, write to PrivateState
    return {"bar": state["foo"] + " is"}

def node_3(state: PrivateState) -> OutputState:
    # Read from PrivateState, write to OutputState
    return {"graph_output": state["bar"] + " Lance"}
	
builder = StateGraph(OverallState,input_schema=InputState,output_schema=OutputState)
builder.add_node("node_1", node_1)
builder.add_node("node_2", node_2)
builder.add_node("node_3", node_3)
builder.add_edge(START, "node_1")
builder.add_edge("node_1", "node_2")
builder.add_edge("node_2", "node_3")
builder.add_edge("node_3", END)

graph = builder.compile()
graph.invoke({"user_input":"My"})
# {'graph_output': 'My name is Lance'}

```


## 2. Node Caching 이란..

캐싱은 어떤 값을 저장하고 재사용하는 것을 의미한다. (특히 시간이 걸리는 계산값을 재계산 없이 재사용) <br>
LangGraph Node는 인풋값을 계산후 아웃풋을 내보내는데, <br>
이 아웃풋 결과를 저장했다가 같은 입력이 들어오면 바로 저장해둔 겨로가를 꺼내쓰는 기능이 있는데.. <br>
이걸 Node Caching이라고 합니다.

Node 정의시, cache_policy를 설정할 수 있다.

```
# 3. 노드에 캐시 정책 설정 (3초 ttl)   ttl = time to live
builder.add_node("expensive_node", expensive_node, cache_policy=CachePolicy(ttl=3))
```
이걸 설정하지 않으면, cache는 소멸하지 않는다.


## 3. Conditional Edges
Edges는 노드간을 연결하는 길이다.<br>
그런데 상황에 따라 이쪽 또는 저쪽으로 선택적으로 연결되길 원할 경우, Conditional Edges를 사용한다.

```
graph.add_conditional_edges("node_a", routing_function)
```

아래는 routing_function 예시
```
def routing_function(state):
    """
    라우팅 함수: 현재 state를 기반으로 다음 노드를 결정합니다.

    state: dict 형태의 상태 정보
    반환: 다음 실행할 노드의 이름(str) 또는 노드 이름의 리스트(list of str)
    """
    # 예시: 상태에 따라 분기 처리
    user_input = state.get("user_input", "").lower()

    if "help" in user_input:
        return "help_node"
    elif "cancel" in user_input:
        return "cancel_node"
    elif "next" in user_input:
        return ["process_node", "log_node"]  # 병렬 실행 (반환값이 문자열 리스트이면 해당 노드들을 병렬 실행)
    else:
        return "fallback_node"
```

routing_function이 True/False 함수인 경우 아래와 같이 다음 노드를 딕셔너리 형태로 맵핑해줄 수 있다.

```
graph.add_conditional_edges("node_a", routing_function, {True: "node_b", False: "node_c"})
```

[Tips]
Use Command instead of conditional edges if you want to combine state updates and routing in a single function.



## 4. Send

일반적으로 LangGraph에서는 모든 **노드(Node)**와 **엣지(Edge)**를 미리 정의합니다.<br>
즉, 어떤 노드들이 있고, 이 노드들이 어떤 순서로 연결되는지를 처음부터 정해놓고 사용하는 방식이에요.<br>
이 방식은 간단한 흐름에는 잘 맞지만, 아래와 같은 경우에는 부족할 수 있습니다:

- 다음에 어떤 노드로 갈지 미리 알 수 없는 경우
- 같은 노드를 여러 번 **다른 상태(State)**로 실행하고 싶은 경우


🧠 예시: Map-Reduce 구조

예를 들어, 하나의 노드가 여러 개의 주제(subject) 리스트를 만든다고 가정해봅시다.

```
state['subjects'] = ['고양이', '강아지', '토끼']
```

그 다음에는 이 각각의 주제에 대해 **개별적으로 농담(joke)**을 만들어야 해요. 즉:

- "고양이" → 농담 생성
- "강아지" → 농담 생성
- "토끼" → 농담 생성

이 때는 총 몇 개의 주제가 나올지 미리 알 수 없기 때문에, 엣지를 고정해 놓을 수 없어요.

🧰 해결 방법: Send 사용

이럴 때 사용하는 게 바로 Send라는 도구입니다.<br>
Send는 다음 두 가지를 정해줍니다:

- 어느 노드로 보낼지 (예: "generate_joke")
- 어떤 상태(State)를 보낼지 (예: {"subject": "고양이"})

아래 코드는 위 내용을 코드로 표현한 것입니다:
```
def continue_to_jokes(state: OverallState):
    return [Send("generate_joke", {"subject": s}) for s in state['subjects']]
```

즉, 주제 리스트에 있는 각 항목마다 generate_joke 노드를 호출하고, 그 주제만 담긴 상태를 보냅니다.<br>
그리고 이 함수를 특정 노드의 **조건부 엣지(conditional edge)**로 등록합니다:
```
graph.add_conditional_edges("node_a", continue_to_jokes)
```
이렇게 하면 node_a 실행 후, subjects에 들어 있는 항목 수만큼 generate_joke 노드가 각각 호출되면서 동작하게 됩니다.



## 5. Command
LangGraph에서는 **그래프 상태(State)**와 **이동 경로(어떤 노드로 갈지)**를 동시에 제어하고 싶을 때가 많습니다. <br>
이럴 때 사용하는 게 바로 Command입니다.


예시로 이해해보기
```
def my_node(state: State) -> Command[Literal["my_other_node"]]:
    return Command(
        update={"foo": "bar"},       # 상태 업데이트
        goto="my_other_node"         # 다음에 갈 노드 지정
    )
```

이 함수는 아래 두 가지 일을 동시에 합니다:
- 상태를 업데이트함 (foo라는 값을 "bar"로 설정)
- 다음에 이동할 노드를 지정함 (my_other_node로 이동)


동적으로 이동하고 싶을 때도 사용 가능합니다. <br>
조건에 따라 다른 동작을 하고 싶을 수도 있죠. 예를 들면:
```
def my_node(state: State) -> Command[Literal["my_other_node"]]:
    if state["foo"] == "bar":
        return Command(update={"foo": "baz"}, goto="my_other_node")
```
여기서는 상태값이 "bar"일 때만 상태를 "baz"로 바꾸고, my_other_node로 이동합니다.

중요한 점!<br>
함수를 만들 때 꼭 **리턴 타입(annotation)**을 아래처럼 지정해줘야 합니다:

```
Command[Literal["my_other_node"]]
```

왜 필요하냐면:<br>
LangGraph가 그래프를 시각화하거나 실행 흐름을 파악할 때 이 노드가 어디로 갈 수 있는지 정확히 알아야 하기 때문이에요.


### Navigating to a node in a parent graph¶
✅ 하고 싶은 일:
서브그래프 안의 노드에서, 부모 그래프의 다른 서브그래프로 **이동(navigate)**하고 싶을 때가 있어요.

🧠 핵심 코드 설명
```
def my_node(state: State) -> Command[Literal["other_subgraph"]]:
    return Command(
        update={"foo": "bar"},
        goto="other_subgraph",  # 이동할 대상은 부모 그래프에 있는 다른 서브그래프
        graph=Command.PARENT    # "부모 그래프로 이동하라"는 표시
    )
```
요점 정리:
- goto="other_subgraph": 어디로 이동할지 지정합니다.
- graph=Command.PARENT: 이동 대상이 부모 그래프에 있다는 걸 알려줍니다.

⚠️ 주의할 점
만약 상태를 업데이트하면서 서브그래프와 부모 그래프 모두 같은 키(예: "foo")를 갖고 있다면,<br>
→ 그 키에 대해 **부모 그래프에서 처리 로직(=reducer)**를 꼭 정의해줘야 해요.<br>
그렇지 않으면 충돌이 생길 수 있어요.

💡 어디에 쓰면 좋을까?
예를 들어, 여러 에이전트(사용자, 챗봇 등)가 함께 일하는 시스템에서<br>
한 명의 작업이 끝나고 **다른 에이전트에게 바통을 넘겨주는 상황(=multi-agent handoff)**에 이 기능이 유용해요.


### 📦 핵심 개념: Handoff (넘겨주기)
🤖 멀티 에이전트 시스템이란?

- 여러 에이전트(작업을 수행하는 주체)가 그래프처럼 연결된 구조로 작동해요.
- 각 에이전트는 어떤 일을 하고 나서, 그 다음에 누가 할지 결정해요.
- 이때 다른 에이전트에게 작업을 넘기는 걸 handoff라고 해요.

🧭 Handoff를 쓸 때 필요한 정보
- destination: 누구에게 넘길지? (에이전트 이름)
- payload: 넘기면서 함께 전달할 데이터 (예: 상태 정보)

🧱 예시: 간단한 handoff 구현
```
def agent(state) -> Command[Literal["agent", "another_agent"]]:
    # 다음 에이전트를 정함 (예: 상황에 따라)
    goto = get_next_agent(...)  # 예: "agent" 또는 "another_agent"
    return Command(
        goto=goto,  # 다음으로 넘길 에이전트
        update={"my_state_key": "my_state_value"}  # 함께 넘길 데이터 (상태 업데이트)
    )
```
이 함수는 현재 상태를 보고, 다음 누구에게 넘길지 결정하고, 상태도 같이 넘겨요.

🔁 복잡한 경우: 에이전트 안에 또 그래프가 있을 때

예를 들어 에이전트 alice와 bob이 있고, 둘 다 내부에 서브그래프(작은 그래프 구조)를 가지고 있다고 해봐요.<br>
alice 내부의 어떤 노드가 bob에게 넘기고 싶을 땐, Command.PARENT를 써서 부모 그래프로 돌아가야 해요.
```
def some_node_inside_alice(state):
    return Command(
        goto="bob",  # bob에게 넘김
        update={"my_state_key": "my_state_value"},
        graph=Command.PARENT  # 부모 그래프에서 bob을 찾아감
    )
```


🧩 시각화를 위한 팁<br>
내부 서브그래프 간 이동을 잘 시각화하려면, 에이전트를 함수로 감싸야 해요.

❌ 이렇게 하지 말고:
```
builder.add_node(alice)
```

✅ 이렇게 해야 함:
```
def call_alice(state) -> Command[Literal["bob"]]:
    return alice.invoke(state)

builder.add_node("alice", call_alice)
```

🛠️ 툴처럼 handoff 사용하기

때때로 에이전트가 어떤 "툴"을 쓰는 것처럼 보이게 할 수도 있어요.<br>
handoff를 툴 함수 안에 넣어서 사용해요.
```
@tool
def transfer_to_bob():
    """bob에게 넘기기"""
    return Command(
        goto="bob",
        update={"my_state_key": "my_state_value"},
        graph=Command.PARENT,
    )
```
이렇게 하면 마치 툴을 호출했는데, 사실은 에이전트 간 제어 흐름을 넘기는 것이에요.

## 6. Runtime Context
💡 무엇을 말하는 건가요?<br>
그래프(graph)를 만들 때 노드에 전달할 추가 정보를 넣고 싶을 수 있어요. 예를 들어:

- 어떤 **LLM 모델(OpenAI, Anthropic 등)**을 쓸지
- DB 연결 정보
- API 키 같은 것들

이런 정보는 그래프의 "상태(state)"랑은 다르니까 따로 전달해야 해요. 이럴 때 사용하는 게 **runtime context**예요.

🧱 1. ContextSchema 만들기
```
from dataclasses import dataclass

@dataclass
class ContextSchema:
    llm_provider: str = "openai"
```

- ContextSchema는 어떤 추가 정보를 보낼지를 정의해요.
- 여기선 기본적으로 "openai"를 사용하는 llm_provider라는 정보를 설정했어요.

🔄 2. 그래프에 context schema 적용하기
```
graph = StateGraph(State, context_schema=ContextSchema)
```

- 이 그래프는 ContextSchema를 기반으로 추가 정보를 받을 준비를 해요.

📤 3. 그래프 실행 시 context 정보 전달하기
```
graph.invoke(inputs, context={"llm_provider": "anthropic"})
```

- 그래프를 실행할 때 context로 "llm_provider": "anthropic"를 넘겨줬어요.
- 즉, 이번에는 OpenAI 대신 Anthropic을 사용하겠다는 뜻이에요.

📥 4. 노드에서 context 값 사용하기
```
from langgraph.runtime import Runtime

def node_a(state: State, runtime: Runtime[ContextSchema]):
    llm = get_llm(runtime.context.llm_provider)
    ...
```

- node_a라는 노드에서는 runtime.context.llm_provider를 통해 "anthropic"이라는 값을 읽어요.
- 그걸 기반으로 어떤 LLM을 쓸지 결정할 수 있어요.

요약 개념 설명
- ContextSchema	그래프 실행 시 전달할 추가 정보의 구조
- context 파라미터	그래프를 실행할 때 실제로 넘기는 값들
- runtime.context	노드 안에서 이 값들을 꺼내 쓰는 방법



## 7. Recursion Limit
LangGraph 같은 시스템에서는 어떤 작업을 순서대로 여러 번 반복할 수 있어요. <br>
이 반복을 **"슈퍼 스텝(super-step)"**이라고 부릅니다.<br>
Recursion Limit은 이런 반복을 최대 몇 번까지 할 수 있는지를 정해주는 숫자입니다.

📌 기본 동작

- 기본으로는 최대 25번까지 반복할 수 있어요.
- 만약 이 한도를 넘으면, GraphRecursionError라는 에러가 발생합니다.

🛠️ Recursion Limit 바꾸기
반복 횟수를 바꾸고 싶으면, .invoke()나 .stream() 함수에 설정을 넣어주면 됩니다.
```
graph.invoke(
    inputs,
    config={"recursion_limit": 5},  # 반복 횟수를 5번으로 제한
    context={"llm": "anthropic"}
)
```
⚠️ 주의할 점
- recursion_limit은 config 딕셔너리의 최상단에 직접 넣어야 해요.
- 다른 설정처럼 configurable 안에 넣으면 안 됩니다.


<<<Config>>>
🔧 설정(config)에는 두 가지 종류가 있어요:
- 일반 설정 값 → configurable 안에 넣어야 함
- 특수 설정 값 → config 딕셔너리의 바깥(최상단)에 직접 넣어야 함

📌 예를 들어 볼게요:
```
graph.invoke(
    inputs,
    config={
        "recursion_limit": 5,              # ✅ 이렇게 직접 넣는 건 OK (특수 설정)
        "configurable": {
            "temperature": 0.7,            # ✅ 일반적인 사용자 설정은 여기!
            "max_tokens": 1000
        }
    },
    context={"llm": "anthropic"}
)
```