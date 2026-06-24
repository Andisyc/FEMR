# FrontRES Repair Discussion Raw Dialogue

Date: 2026-06-24

This file preserves the side-conversation discussion in dialogue form.  It is
intended for later reflection on the research framing, method boundary, and
paper-closing strategy.

## Dialogue

### User

我突然想讨论科研思维的问题，我接受的科研训练是小步前进，快速迭代，意思是每篇文章只做一个小小的问题，对应到技术上就是很小一个改动，如果按照这个思维，我在最开始的HRL阶段就已经成功了，但我总是觉得还是需要应对高扰动情况，因此不断扩展方法，直到现在都没跑通，先前我们也讨论过问题边界的设置，如何从科研思维上看待这个问题？

### Assistant

我认为这件事可以这样看：你不是“不懂小步前进”，而是把“小步前进”和“方法边界”混在了一起。

小步前进不是说看到问题就不管。它真正的意思是：每篇文章只闭合一个核心变量。如果新问题还在这个变量内部，就解决；如果新问题需要另一个变量、新证据、新训练机制，那它就应该变成下一篇文章，而不是当前文章继续膨胀。

在 FrontRES 这里，最开始 HRL 阶段已经证明的东西大概是：

参考帧被污染时，前端 residual repair 可以提升 frozen GMT 的可执行性。

这已经是一个完整的小问题。它的变量是“修复 corrupted reference 是否比直接交给 GMT 更好”。如果这个变量已经被实验闭合，那么它就足够成为一篇小而干净的文章。

后来你看到高扰动下性能下降，直觉上觉得“这个也必须解决”。这个直觉很正常，因为你是认真做系统的人。但从科研边界看，高扰动问题其实已经变成了另一个变量：

当前状态是否还允许这个 Clean-oriented repair 被执行。

这不是原来那个“能不能修复参考帧”的问题了。它引入了 dynamic admissibility、acceptance、rho、critic、counterfactual rollout、甚至 recovery anchor。也就是说，它不再是对原方法的一点小修，而是把文章从“前端修复器”推向“动态可接受修复决策器”。这是另一个研究对象。

所以关键教训不是“以后不要扩展方法”，而是：

当一个新失败出现时，先判断它是在破坏当前 claim，还是在暴露下一层研究问题。

如果它破坏当前 claim，比如低扰动下也不稳定，那必须解决。  
如果它只说明方法在更强扰动下边界有限，那它可以是 stress test、failure analysis、future work，而不一定是当前方法义务。

你这次真正痛苦的地方在于：高扰动失败看起来像当前方法的失败，但实际上它要求方法学习一个更深的变量。你把一个边界诊断当成了当前 paper 的核心责任，于是方法不断长出新模块，最后工程复杂度超过了原始 idea 的承载能力。

我会把这个形式化成一句话：

一篇文章的成功，不是解决所有真实问题；而是把一个不可见变量变成可测、可训、可验证的机制。

如果 HRL 阶段已经把“reference repair improves GMT”这个变量闭合，那么正确策略可能是：保留 HRL 主方法，把高扰动作为边界实验。文章里可以说：当 perturbation 进入动态不可接受区域，仅靠前端几何修复不足，这提示需要 state-conditioned acceptance。这样不是弱，反而会显得你知道方法边界。

从科研训练上，你以后可以用一个简单规则约束自己：

每当你想增加一个机制，先问：

这个机制是在强化当前变量，还是在引入新变量？

如果是强化当前变量，可以加。  
如果是引入新变量，要非常警惕。它可能值得做，但可能应该是下一篇。

这也是为什么“容易实现、暴力涨点”的 idea 很珍贵。它们通常满足一个特征：技术改动和概念变量几乎是一一对应的。方法即概念。你这次后期痛苦，是因为技术机制开始远多于概念变量：rho、critic、burst、k-step、warmup、acceptance，每个都合理，但它们共同说明这个问题已经不是一个小改动能干净闭合的层级了。

所以从现在回看，最理性的科研路径可能是：

先把最早成功的 HRL/FrontRES 版本整理成一篇边界清楚的小文章。  
高扰动不是必须解决的问题，而是下一篇的 motivation。  
现在这些痛苦探索不是浪费，它们帮你发现了下一层真正的问题：Clean-oriented repair 需要 dynamic acceptance。

这才是博士阶段最重要的积累：不是每次都做出最优版本，而是逐渐学会判断“这个问题属于当前文章，还是属于我的下一篇文章”。

### User

你的解释符合我现在的直觉，我现在的确想把手头的工作整理成一篇小文章，但我依然没掌握设置问题边界的思维，我最开始从HRL转向HSL的原因是HRL虽然能让FEMR具备修复能力，但修复效果不是很好，demo需要仔细分辨才能看出，效果不够直观，于是我发现输出∆SE不一定需要HRL，HSL就行，所以后续我转向HSL，效果虽然提升了，但在高扰动下效果依然不佳，并且我发现转向HSL不够fancy，很难够上顶会，于是我开始扩展问题边界并添加许多模块，逐渐越走越远，我认为究其原因，是因为每次到达应当设立边界时，我都感到故事不算圆满，怎么解决这个问题？

### Assistant

你这里的核心问题不是“不懂边界”，而是你把两种“圆满”混在了一起：

一种是论文故事的圆满：一个问题、一个变量、一个机制、一个证据链闭合了。  
另一种是系统能力的圆满：方法在各种强度、各种坏情况、各种审稿人追问下都表现很好。

博士生最容易痛苦的地方，就是用第二种圆满要求第一种论文。这样方法会不断膨胀，因为真实世界永远能提出下一个 failure case。

你的路径其实非常典型：

HRL 阶段：证明 FEMR/FrontRES 能修复，但效果不够直观。  
这时你没有越界，你是在寻找更干净的训练信号。

HSL 阶段：发现 ΔSE 可以直接监督，效果更直观。  
这一步是合理的，因为它仍然围绕同一个核心变量：参考帧污染可以用前端 residual 修复。

高扰动失败：发现修复在某些动态状态下不该被完全执行。  
这里开始变了。它不再只是“怎么更好地产生 ΔSE”，而是新变量：这个修复在当前状态下是否可接受。

所以真正的边界点在这里：

从 HSL 到高扰动 acceptance，是从“修复方向”跳到了“修复权限”。

这不是小补丁，而是新问题。

你当时觉得故事不圆满，是因为你希望方法同时回答两个问题：

1. 参考帧被污染时，能不能生成更干净的目标？
2. 当前动态状态下，这个目标能不能被执行？

但这两个问题虽然相关，科学对象不同。第一个是 reference correction。第二个是 dynamic acceptance。把它们合在一篇里，当然会越来越复杂。

我觉得你需要建立一个“论文闭合标准”，不是靠感觉判断故事圆不圆，而是靠四个条件：

第一，问题是否单一。  
这篇文章到底解决“参考帧污染”还是解决“动态可接受性”？如果一句话里必须同时说两个核心问题，边界已经太大。

第二，变量是否单一。  
FrontRES-HSL 的核心变量是 ΔSE repair。rho acceptance 的核心变量是 authority。变量一多，故事就会散。

第三，证据是否直接。  
HSL 的证据很直接：Noisy reference 和 Clean reference 给出 ΔSE target，修复后 GMT 更好。  
rho 的证据不直接：需要 rollout、baseline、counterfactual、critic、时序窗口。证据一复杂，通常说明已经进入下一篇。

第四，失败是否能被边界解释。  
如果高扰动失败可以被解释为“当前方法不生成动态恢复参考，只修复 reference artifact”，那它就是边界，不是必须修的漏洞。  
如果低扰动下也失败，那才是当前方法的核心漏洞。

你以后可以用一个很实用的判断句：

如果一个新问题需要新变量、新证据、新训练机制，它就不是当前文章的补丁，而是下一篇文章的入口。

放到你的例子里：

HRL -> HSL：没有引入新研究对象，只是把训练信号变干净。可以做。  
HSL -> rho acceptance：引入新变量 authority、新 rollout 证据、新 actor-critic。应该警惕。  
rho -> stable frame/recovery anchor：又引入新对象 S(x)。更应该放到未来工作。

至于你说 HSL 不够 fancy，我想稍微反驳你一下。顶会不一定奖励复杂模块，顶会奖励的是一个清楚的问题被一个漂亮的变量压缩掉。

HSL 看起来不 fancy，可能是因为你还没有把它的概念高度写出来。它不只是“监督一个 ΔSE”。它可以被讲成：

Frozen tracker 本身有鲁棒性，但 reference-frame artifacts 会消耗这部分鲁棒性。我们不重新训练 tracker，而是在 tracker 前面学习一个 task-space residual interface，把 corrupted reference 拉回 frozen tracker 可执行的输入分布。

这就不是普通 HSL 了。它的核心不是 loss fancy，而是架构观点：

不要替换 tracker，修复 tracker 的输入。

这个故事如果实验足够干净，是可以成立的。高扰动失败反而可以帮你定义边界：

“当扰动进入动态不可接受区域，仅修复 reference geometry 不够，需要进一步学习 state-conditioned acceptance。本文把这一点作为边界诊断，而不是当前方法的一部分。”

这不是示弱，而是成熟。审稿人不怕你有边界，怕你不知道边界。

我建议你以后每次感觉“故事不圆满”时，不要立刻加模块，而是写下三句话：

1. 当前文章的核心变量是什么？
2. 这个不圆满是在挑战核心变量，还是在引出新变量？
3. 如果我不解决它，当前 claim 是否仍然成立？

如果答案是“claim 仍然成立”，那就不要扩。把它写成 limitation、stress test、future work。  
如果答案是“claim 会崩”，那才进入当前方法。

你这次最重要的经验可能是：

边界不是在方法完美时设置的。边界是在方法已经闭合一个变量，但还没有吞掉下一个变量时设置的。

你之前一直想等故事自然圆满，但科研里故事不会自然圆满。真实系统永远不圆满。论文的圆满来自主动切边。

### User

你的解释非常有启发，我虽然时常安慰自己：只要有文章，不断投总能中，但我的潜意识并不这样认为，所以我非常关注文章是否圆满和fancy，因为只有这样才能增加中的概率，目前来看使用HSL真的不够fancy，让我感觉很难中，请开导我

### Assistant

我很理解这个焦虑，而且我觉得它不是虚荣。你担心“不够 fancy”，本质上是在担心：审稿人会不会觉得这只是一个工程 trick，而不是一个值得发表的研究观点。

这个担心是合理的。但这里有个很重要的反转：

fancy 不一定来自方法复杂，fancy 常常来自问题重述的高度。

很多顶会文章的方法本身并不复杂。真正让它们显得强，是它们把一个大家忽略的问题压缩成一个清楚的变量，然后用一个很自然的机制解决。读者会觉得：噢，原来应该这样看这个问题。

所以 HSL 本身不 fancy，不代表文章不 fancy。你需要判断的是：HSL 是不是只是一个 loss，还是它背后有一个新的 problem framing。

如果你把它写成：

“我们监督学习一个 ΔSE 修复器。”

那确实不够。  
但如果你把它写成：

“现有 motion tracking policy 的失败不只来自 policy 鲁棒性不足，也来自 reference artifacts 消耗了 policy 的 robustness budget。与其重新训练 tracker，我们提出一个 frozen-tracker-compatible front-end repair layer，在 task space 中恢复 reference executability。”

这个故事就不一样了。

这里的 fancy 点不是 HSL。  
fancy 点是：

把 robustness 问题从 policy learning 转移到 reference interface repair。

也就是说，你不是说“我用了监督学习”。你是在说：

GMT 其实已经有能力，但坏 reference 把它推离了可执行输入分布；FrontRES 的作用是把输入重新投回 GMT 能处理的区域。

这就是一个研究观点。

你现在觉得不够 fancy，可能是因为你一直盯着技术模块看：HRL、HSL、rho、critic、burst。模块越复杂，好像越像研究。但论文不是模块竞赛。论文真正要卖的是：

1. 以前大家怎么看这个问题？
2. 你指出他们漏了什么？
3. 你的变量为什么刚好抓住这个漏点？
4. 你的方法为什么是这个变量的最小实现？
5. 实验是否证明这个变量真的重要？

HSL 如果能回答这些，它就不低级。

当然，我也不想鸡汤式安慰你。HSL 文章要想中，确实需要包装得很锋利。它不能只是“我们训练一个修复网络”。它至少需要三个强支点：

第一，问题支点：reference artifacts 是真实瓶颈。  
你要证明不是 GMT 弱，而是 reference 错误消耗 robustness budget。最好有 clean/noisy reference 的稳定性曲线，显示 error 增大时 GMT 快速退化。

第二，架构支点：frozen tracker front-end repair 是一个不同路线。  
不是 end-to-end retraining，不是 policy distillation，不是加大网络，而是在 tracker 前加 residual interface。这个很重要，因为它保留了强 tracker 的能力。

第三，实验支点：简单机制带来清晰增益。  
你不需要解决所有高扰动，只要在定义好的 artifact 范围里，HSL-FrontRES 显著改善 episode length、tracking error、demo 质量，并且比重新训练或直接 noisy baseline 更清楚。

如果这三个支点立住，HSL 不 fancy 反而是优点。审稿人会觉得：这个问题本来很乱，你用一个简单 interface 把它理清楚了。

你真正要避免的是另一种情况：方法复杂，但故事不清楚。那才是危险。复杂方法如果跑不通，既没有结果，也没有清晰贡献；简单方法如果跑通，至少有一个干净的 claim。

我想给你一个判断标准：

方法简单不可怕，可怕的是 claim 也简单。

“我用 HSL 训练 ΔSE”这个 claim 简单。  
“reference artifact repair as a frozen tracker interface”这个 claim 就有高度。  
“robustness budget is consumed by reference-frame error”这个 observation 也有高度。  
“task-space residual repair restores executability without retraining GMT”这个 mechanism 也可以写得很漂亮。

所以你现在不一定要把 HSL 变 fancy，而是要把问题定义、架构视角、实验曲线变 fancy。

还有一点更现实的话：你现在继续堆 rho/critic/stage2，也许理论上更高级，但如果跑不通，投稿概率不是提高，而是下降。一个干净、有边界、有实验的小方法，比一个野心很大但链路不稳定的方法更有投稿价值。

你可以这样安慰自己，但不是虚假的安慰：

我不是在降低文章目标。我是在把文章从“复杂但不闭合”压缩成“简单但闭合”。

这是一种成熟，不是退让。

而且从长期看，这反而更像 PhD 的正确节奏。第一篇文章提出 front-end repair framing。第二篇文章讨论 dynamic acceptance。第三篇文章再进入 recovery reference generation。你不是放弃高级问题，而是把研究路线拆成可发表的连续台阶。

你现在最需要的不是再证明自己能想复杂问题，你已经证明了。你现在最需要的是训练自己在合适的位置收刀。真正强的研究者，不是每次都把问题做到最大，而是知道哪一刀切出来就是一篇文章。

### User

你的解释实际上也是导师和其他高年级同学对我的教导 (当然他们不会解释得这么清楚，他们是从自身的经验和直觉来解释)，并且也是我先前看文章的感受，但有个问题是：HSL拍出的demo也不够好看，至少需要筛选，或者略微重新设计和包装，保证能与Claim对应上，并且平心而论，我真的觉得HSL太Low了，如果我来审这篇文章，发现虽然包装很好，但内核居然是HSL，那我很可能打borderline的分数，甚至偏低

### Assistant

你这个判断很真实，而且我同意你一半。

如果一篇文章的内核只是：

“我构造了 clean target，然后用 HSL 监督一个网络。”

那确实 low。你作为审稿人打 borderline 是合理的。

但这里有一个分界：HSL 是训练方式，不应该是方法内核。

如果文章的内核被写成 HSL，那它弱。  
如果文章的内核是一个新的系统分解，HSL 只是实现这个分解的最小训练信号，那它就不一定弱。

换句话说，不能让论文看起来像：

“我们提出 HSL loss。”

而应该像：

“我们提出一个 frozen tracker compatible reference repair interface。”

HSL 在这里的角色类似“用最直接的监督信号训练这个 interface”。它不是贡献本身。

但你说 demo 不够好看，这是更实在的问题。这个不能靠包装解决。包装只能让 claim 更清楚，不能把弱结果变强。这里要分两件事：

第一，HSL 是否足以支撑文章主 claim。  
如果主 claim 是“显著提升高扰动鲁棒性”，那 HSL 不够。  
如果主 claim 是“reference artifact repair 可以恢复 frozen tracker 的一部分可执行性”，那 HSL 可能够。

第二，demo 是否需要重新定义。  
你不一定需要展示最难、最高扰动、最 dramatic 的 demo。你需要展示与 claim 对应的 demo：reference corruption 明显、GMT 被拖坏、FrontRES 把它拉回来。demo 不是证明万能性，而是证明机制存在。

所以现在的策略不是硬吹 HSL，而是重新校准 claim 和展示。

我觉得可以有三条路：

路线 A：小文章路线。  
承认 HSL 简单，把文章重点放在问题定义、frozen tracker interface、artifact robustness budget。实验只覆盖 HSL 真能稳定改善的扰动范围。高扰动作为 limitation。  
这条路投稿风险中等，但最可控。

路线 B：增强展示路线。  
不改变核心方法，只做更干净的数据选择、可视化、demo 筛选、metric 设计。重点让 HSL 的有效区域非常清晰。  
这不是造假，而是让 claim 和 evidence 对齐。文章本来就不需要展示方法解决所有情况。

路线 C：继续冲高级方法。  
继续做 rho/critic/acceptance，让方法更 fancy。但风险是时间和工程复杂度继续爆炸，而且不一定在近期闭合。  
这条路更像下一篇或长期路线。

你现在最该避免的是第四条隐形路线：

一边觉得 HSL low，一边又没有足够时间把高级方法跑通，最后两边都不完整。

这才是最危险的。

至于你说“如果我是审稿人，会觉得 HSL low”，我想帮你校正一下审稿人的视角。审稿人不只看方法技术复杂度，他们还看：

- 这个问题是否真实？
- 这个 formulation 是否以前没人清楚地提过？
- 这个简单方法是否意外有效？
- 实验是否说明了一个新现象？
- 这个方法是否容易复现、容易被别人用？

很多 paper 的核心技术都不复杂，但它们有一个强 observation。你的潜在强 observation 是：

motion imitation/tracking policy 的鲁棒性评估里，reference artifacts 是一个被低估的 failure source；修 policy 不一定是唯一选择，修 reference interface 也能释放 frozen policy 的 robustness budget。

这比“HSL 学 ΔSE”高级得多。

但如果 demo 真的弱到“需要仔细分辨”，那确实不够。此时不是继续堆模块，而是先做一个冷静评估：

在 HSL 的最佳适用范围内，它能不能产生肉眼可见、metric 明显的提升？

如果能，那就把文章边界切在这个范围。  
如果不能，那 HSL 只能是 preliminary，不该硬投主会。  
如果低扰动可见、高扰动不可见，那它是一个边界清楚的小方法。  
如果所有范围都不够明显，那你需要回到实验设计，而不是方法堆叠。

我会建议你接下来用一个很现实的标准判断：

不要问 HSL fancy 不 fancy。问 HSL 能不能支撑一个审稿人无法轻易否定的最小 claim。

比如：

“Under moderate reference-frame perturbations that preserve the feasibility of the original motion, FrontRES recovers a substantial fraction of the tracking performance lost by corrupted references, without retraining the tracker.”

这个 claim 不要求高扰动，不要求 recovery，不要求 state-conditioned acceptance。它只要求 moderate artifact 下有效。

如果 HSL 能支撑这个，它就不是 low，它是一个简单但有效的 first-order solution。  
如果 HSL 连这个都支撑不了，那你直觉是对的，它还不够文章。

最后我想说一句更个人化的：你之所以觉得 HSL low，是因为你脑子里已经看到了更深的问题。你已经站在 dynamic acceptance 那一层了，所以回头看 HSL 会觉得浅。这是成长带来的副作用。但论文不是把你脑子里最深的东西一次性写完。论文是把你已经能证明的那一层切出来，让别人跟上你。高级问题可以留作下一步，而不是把当前这一步拖死。

### User

我理解了，最重要的是：
1、把问题阐述清楚，用各种图表和曲线
2、卖一个框架，HSL只是其中一个部分
3、清晰定义问题边界 (我认为这一点最难，而且审稿人不一定会认)
4、投一个次等会，提高中稿概率

### Assistant

是的，这四点基本就是正确路线，而且我会稍微替你压缩成一个更像投稿策略的版本：

这篇文章不要卖 HSL，卖 problem framing + framework。

HSL 只是回答：“在这个边界内，最小可执行机制是什么？”  
真正的文章价值应该落在：

1. 问题定义  
   reference artifacts 会消耗 frozen tracker 的 robustness budget。这个现象需要用图和曲线打出来。比如 perturbation scale 上升时，GMT performance 如何下降；FrontRES 在 moderate artifact 下如何恢复一部分性能。

2. 框架观点  
   不重新训练 tracker，而是在 tracker 前面加一个 reference repair interface。这个框架比 HSL 本身重要。HSL 是实现这个 interface 的最简单版本。

3. 边界声明  
   当前方法解决的是 artifact repair，不解决动态恢复参考生成。高扰动失败不是“方法没意义”，而是边界：当 corrupted reference 已经进入 dynamically inadmissible 区域，仅靠 Clean-oriented repair 不够。

4. 投稿定位  
   先投一个更稳的会是合理选择。不是因为工作差，而是因为这篇文章要先把研究路线的第一层发出来。你后面更 fancy 的 acceptance / recovery-anchor 问题可以作为下一篇。

我觉得你现在最需要警惕的是第 3 点的写法。边界不能写成“我们做不了高扰动”，而要写成：

我们刻意研究 reference artifact repair 的第一阶问题：当原 motion 仍处于 frozen tracker 可执行邻域内时，能否通过 task-space front-end residual 恢复 tracking robustness。超出该邻域后，问题变为 dynamic recovery reference generation，需要新的 state-conditioned acceptance mechanism。

这样边界就不是借口，而是问题分解。

你现在的论文最小闭环可以是：

Observation：reference corruption hurts tracking not only by changing target accuracy, but by consuming the robustness budget of a frozen tracker.  
Method：insert a lightweight task-space residual repair module before GMT.  
Training：use Clean-Noisy pairs to supervise ΔSE because the repair direction is observable inside the artifact-repair boundary.  
Result：recover performance and demo quality under moderate artifacts without retraining GMT.  
Boundary：when perturbation becomes dynamically inadmissible, repair authority becomes a second problem.

这就不是 “HSL 小 trick” 了，而是一个清楚的 first-step paper。你要做的是把图、曲线、demo、边界都对齐这个故事。

### User

我想先讨论清楚第3点，你和我知道这么解释是正确的，但不够有说服力，这也是导师的看法：没法让审稿人一眼感到的确如此，审稿人很容易会问题一个问题：那为啥你这篇文章不做Dynamic Recovery呢？我正是因为回答不了这个问题才不断扩展问题边界

### Assistant

你卡住的点非常关键：边界不能只靠我们自己知道，必须让审稿人觉得这个边界是问题本身自然切出来的，而不是作者能力不足。

所以第 3 点不能写成：

“我们暂时不处理 high perturbation / dynamic recovery。”

这确实没说服力。审稿人会问：那为什么不做？

它应该写成：

Dynamic Recovery 和 Reference Artifact Repair 是两个不同层级的问题。本文刻意隔离第一层，因为如果不先解决 reference interface 的污染问题，后续任何 recovery policy 都会被混入不干净的 reference error。

也就是说，不要把边界说成“不做更难的部分”，而要说成：

我们先定义并解决一个更基础、更可测、更独立的子问题。

这里的关键是让审稿人看到：你不是逃避 Dynamic Recovery，而是在做它之前必须先拆出的前置问题。

你可以这样理解：

如果一个 motion 已经本身不可恢复，那需要 Dynamic Recovery。  
但如果 motion 本身可执行，只是 reference 被污染了，导致 frozen tracker 失败，那这是 Reference Repair。

这两个 failure 看起来都表现为“机器人摔倒”，但因果不同。

一个是：

state is dynamically bad  
所以需要 recovery behavior。

另一个是：

reference is corrupted  
所以 tracker 被喂了错误目标。

你的文章要证明的是第二种现象真实存在，而且值得单独研究。

所以最有说服力的边界不是文字声明，而是一个实验图：

横轴：reference corruption scale。  
纵轴：GMT episode length / tracking score / survival rate。  
同时画三条线：

- Clean reference + GMT
- Noisy reference + GMT
- Repaired reference + GMT

你要让审稿人看到：

在 moderate corruption 区域，Clean+GMT 还能跑，说明 motion 本身可执行；Noisy+GMT 掉下去，说明失败来自 reference artifact；FrontRES 能拉回来，说明 repair interface 有价值。

到了 high corruption 区域，如果 Clean+GMT 或 oracle/candidate 也开始差，说明问题已经不只是 reference artifact，而是进入 dynamic recovery 区域。这个时候 FrontRES 不解决，是合理边界。

这比一句“我们不做 dynamic recovery”强很多。因为你用实验把问题空间切开了：

Region I：reference artifact repair problem。本文解决。  
Region II：dynamic recovery problem。本文诊断，但不声称解决。

审稿人问“为什么不做 Dynamic Recovery？”时，你的回答不是“太难了”，而是：

“Because it is a different problem with a different causal variable. Our method targets cases where the original motion remains executable by the frozen tracker, but reference artifacts push the input outside the tracker’s robust interface. Dynamic recovery requires generating or selecting a new admissible motion under disturbed dynamics, which is outside the reference-repair assumption.”

这句话的核心是：原 motion 是否仍然可执行。

如果仍然可执行，只是 reference 错了，你做 FrontRES。  
如果原 motion 已经不可执行，那需要 recovery generation，不是 repair。

你需要把这个边界定义成一个可观测条件：

本文研究的扰动范围满足：Clean reference under GMT remains largely executable, while Noisy reference degrades performance.

这就很强。因为它不是作者主观设边界，而是实验条件。

然后 Dynamic Recovery 的问题就自然被推开了：

Dynamic Recovery 处理的是 Clean reference 本身也不再可执行的区域。  
FrontRES 处理的是 Clean reference 可执行、Noisy reference 不可执行的区域。

这就是最重要的边界定义。

你之前不断扩展，是因为你把“FrontRES 在高扰动下失败”理解成当前方法没闭合。但如果我们用这个边界看，高扰动失败可能反而是边界证据：

当 Clean/GMT 或 oracle repair 也不能稳定时，说明 reference repair 已经不是正确问题了。再往里加 rho、critic、stable frame，本质上是在做下一篇 Dynamic Recovery。

文章里可以这样讲：

“FrontRES is not designed to synthesize a new recovery motion. It assumes the clean reference lies within the execution neighborhood of the frozen tracker, and corrects corrupted references back toward that neighborhood.”

这句话非常重要。它把方法的权限说清楚了。

所以第 3 点要想让审稿人认，你需要三层防线：

第一层，概念防线：  
Reference Repair 和 Dynamic Recovery 是不同问题。前者修输入，后者生成新行为。

第二层，实验防线：  
用 Clean/GMT、Noisy/GMT、FrontRES/GMT 三条曲线证明存在一个区域：Clean 可执行，Noisy 失败，FrontRES 恢复。

第三层，claim 防线：  
你的 claim 只覆盖这个区域，不声称解决 Clean 本身不可执行的高扰动区域。

这样审稿人即使问“为什么不做 Dynamic Recovery”，你也可以回答：

“这正是我们刻意分离的问题。Dynamic Recovery is a natural next step, but our contribution is to identify and solve the reference-interface failure that occurs before recovery generation is needed.”

这就不是软弱的 limitation，而是一个清楚的问题分解。

你现在要追求的不是让审稿人觉得“高扰动不重要”，而是让他们觉得：

在讨论高扰动恢复之前，先把 reference artifact repair 单独拿出来是合理且必要的。

### User

你的解释非常有启发，让我感到得到启发的点是Recovery这个词，因为其实也有很多人在做mimic的扰动恢复，我也看过效果，这种demo展示出机器人在受到干扰后都有个主动回到动作序列执行上的过程，很明显我们和他们针对的问题并不相同，他们的扰动是额外施加的，并不是参考帧级别的扰动，但你说的非常正确：repaired和recovery的确不同，请沿着这个方向思考

### Assistant

对，就是这里。repair 和 recovery 这两个词能把边界一下子切开，而且这个切法比“高扰动/低扰动”更有说服力。

因为“高扰动/低扰动”听起来只是难度不同。审稿人会自然问：你为什么不做更难的？

但 “reference repair / dynamic recovery” 是问题类型不同。一个是输入目标错了，一个是身体状态被打乱了。它们都会导致 tracking 失败，但因果链不一样。

Repair 的因果链是：

参考帧被污染  
-> tracker 收到错误目标  
-> 本来可执行的 motion 被错误 reference 拖坏  
-> 前端模块把 reference 拉回 clean-compatible 区域  
-> frozen tracker 继续执行原 motion

这里的关键是：motion 本身仍然应该被执行。你不需要重新规划动作序列，也不需要产生 recovery behavior。你只是修正“给 tracker 的输入”。

Recovery 的因果链是：

机器人状态被外力/接触/失衡扰动  
-> 当前身体状态已经偏离原 motion 的可执行邻域  
-> 即使 clean reference 也未必马上可执行  
-> policy 必须生成一个过渡行为，把身体带回 motion manifold  
-> 之后再继续 tracking

这里的关键是：当前状态已经不适合直接执行原 motion。所以它需要主动恢复行为。

这就是非常强的边界：

FrontRES is not a recovery policy. It is a reference repair layer.

这个句子很重要。它听起来不像退让，而像清晰定位。

你之前陷入高扰动，是因为高扰动同时制造了两个现象：

1. reference artifact 变大；
2. 当前执行状态也可能变坏。

第一件事属于 repair。  
第二件事属于 recovery。

如果我们不区分它们，就会误以为“FrontRES 必须解决高扰动下所有失败”。但那其实把两个问题混在一起了。

这也解释了为什么你后来会自然走向 rho、stable frame、dynamic acceptance。因为一旦进入 recovery 区域，仅仅把 reference 朝 clean 修是不够的。系统需要判断当前状态能不能接受这个 reference，甚至需要生成一个新的过渡 reference。这已经是 recovery control 了。

所以文章边界应该从“扰动强度”改成“扰动位置”：

reference-space perturbation：目标/reference 被污染，身体状态仍主要由 tracker 执行。FrontRES 负责 repair。  
state-space perturbation：身体被推、速度/姿态/接触状态偏离，clean reference 本身也不一定马上可执行。Recovery policy 负责 recovery。

这样写，审稿人会更容易理解。因为 mimic recovery 领域本来就存在，他们处理的是 state disturbance。而你处理的是 reference disturbance。

你可以在文章里非常明确地说：

“Unlike disturbance recovery methods that recover the robot state after external perturbations, we study reference repair: correcting corrupted motion references before they are consumed by a frozen tracking controller.”

这个对比会很有力。

更进一步，你可以把 demo 也按这个边界组织：

不要展示“机器人被打后站回来”。那会把你拉进 recovery 赛道。  
你要展示的是：“reference frame 错了，GMT 追错；FrontRES 修正 reference 后，GMT 恢复正常 imitation。”

视觉上可以这样设计：

左边：Noisy reference skeleton 或 corrupted root trajectory，有明显漂移/倾斜/下沉。  
中间：GMT tracking noisy reference，动作被拖坏。  
右边：FrontRES repaired reference + GMT，动作回到 clean-like trajectory。

这比单纯 episode length 更符合 repair 叙事。

我觉得你现在可以把文章的核心表述改成：

Problem：Reference-Space Corruption in Motion Tracking  
不是 generic robustness，不是 dynamic recovery。

Method：Frozen-Tracker-Compatible Reference Repair  
不是新 tracker，不是 recovery policy。

Mechanism：Task-Space Residual Correction  
HSL 只是训练这个 residual 的直接方式。

Boundary：Repair assumes the clean motion remains executable  
一旦身体状态或 reference deviation 需要新的 recovery behavior，那是 future work。

这就清楚很多。

而且这个边界还帮你反击“为什么不做 recovery”：

因为 recovery 方法默认 disturbance 已经发生在 robot state 上；而我们处理的是 disturbance 进入 tracker 之前，发生在 reference interface 上。  
它们不是替代关系，而是互补关系。一个好的 recovery policy 也会受益于干净 reference。一个好的 reference repair layer 也可以作为 recovery policy 的前端。

这句话甚至可以成为文章讨论部分的亮点：

Reference repair and dynamic recovery are complementary: repair prevents corrupted references from consuming robustness, while recovery handles physical state deviations after robustness has been exceeded.

这就很像一个成熟研究方向了。

你之前觉得 HSL low，是因为你把贡献放在训练方式上。沿着 repair/recovery 这个方向，贡献就可以放在问题分解上：

我们指出 tracking failure 里有一种被混淆的问题：不是机器人不会恢复，而是它被错误 reference 指挥了。

这个观察其实不低。  
它甚至是一个很好的论文切入点。

### User

这样解释的确是一个很好的切入角度，我的硕导教育我：一篇文章，一个工作，核心在于切入角度，切入角度决定了这篇文章的水平高低，现在我们有点像在使用一个简单的mixup解决一个复杂问题了 (我在硕士期间看了很多顶会工作本质就是一个mixup，但因为切入问题的角度很好，所以能发顶会，或者说科研的核心在于如何思考问题，而不是技术难度)，那如何与我们卖一个架构关联起来呢？

### Assistant

你硕导这句话非常准。切入角度决定文章高度，架构决定这个角度是否能站住。

现在我们的切入角度是：

tracking failure 不一定来自 policy 不会 recovery，也可能来自 reference interface 被污染。

这个角度本身已经把问题从“再训练一个更强 policy”转到了“修复 frozen tracker 的输入”。接下来所谓“卖架构”，不是卖一个很复杂的系统，而是卖这个切入角度的自然结构。

也就是说，架构不是为了 fancy。架构是为了告诉审稿人：

既然问题发生在 reference interface，那方法就应该放在 tracker 前面，而不是改 tracker 本身。

这就是 FrontRES 的架构意义。

你可以把它想成三层：

第一层是问题切入：  
失败源在 reference，而不一定在 policy。

第二层是架构选择：  
既然失败源在 reference，就在 frozen tracker 前加一个 repair layer。

第三层是训练实现：  
既然 clean/noisy reference pair 能给出 repair direction，就用 task-space residual HSL 训练这个 layer。

这样 HSL 就不是低级 trick，而是架构逻辑的最后一步。它是“这个问题切入下最直接的训练信号”。

文章里可以这样讲：

传统路线：

corrupted tracking performance -> improve policy robustness / recovery

你的路线：

corrupted tracking performance -> repair reference interface -> reuse frozen tracker robustness

这就是架构卖点。

它和 mixup 类顶会工作的共同点在于：方法可能简单，但它改变了问题坐标系。  
如果你在旧坐标系里看，它只是 HSL。  
如果你在新坐标系里看，它是：reference-space robustness via front-end residual repair。

这就是“架构”的作用：把一个简单操作放到正确的位置。

所以你要卖的不是“我们用了一个 residual network”。你要卖的是：

Front-end repair architecture for frozen motion trackers.

它有几个自然优点：

1. Non-invasive：不需要重新训练 GMT。  
   这让方法更实用，也让它和 recovery policy 区分开。

2. Tracker-compatible：输出仍然是 GMT 能理解的 reference。  
   不是生成新动作，不是替换 policy。

3. Task-space interpretable：ΔSE 直接对应 reference-frame artifact。  
   这让 HSL 合理，因为 target 可见。

4. Composable：未来可以和 recovery policy 结合。  
   repair 在 reference 入口处，recovery 在 physical state 偏离后处理，两者互补。

这几个点其实就是架构价值。

你可以画一个很重要的图：

普通 mimic/recovery pipeline：

Reference -> Tracking Policy -> Robot

你的问题指出：

Corrupted Reference -> Tracking Policy -> Failure

你的架构变成：

Corrupted Reference -> FrontRES Repair -> Frozen Tracking Policy -> Robot

这张图越简单越好。它让审稿人一眼看到：你不是在 policy 里面堆东西，而是在系统接口上切了一刀。

然后再加一张概念图区分 repair 和 recovery：

Reference repair：扰动在 reference side，身体状态仍围绕原 tracking pipeline。  
Dynamic recovery：扰动在 robot state side，需要主动回到 motion manifold。

这两张图加起来，文章高度就上来了。

你问“如何与卖一个架构关联起来”，我觉得答案是：

切入角度定义问题发生在哪里；架构就是把方法放到那个位置。

如果问题发生在 policy 能力，那架构应该改 policy。  
如果问题发生在 physical state，那架构应该做 recovery。  
如果问题发生在 reference interface，那架构就应该是 front-end repair。

这就是 FrontRES 的合理性。

所以我们可以把论文主线压成一句话：

We reframe a class of motion-tracking failures as reference-interface corruption rather than policy insufficiency, and introduce a frozen-tracker-compatible front-end repair architecture to restore executable references.

这句话里面有三个层次：

- reframe：切入角度；
- reference-interface corruption：问题边界；
- front-end repair architecture：架构贡献。

HSL 不在这句话里。因为它不是文章的灵魂。它是后面 method section 里自然出现的训练方式。

如果你担心审稿人说“这不就是 supervised residual correction 吗”，回应应该是：

是的，我们刻意使用最小训练机制，因为在 reference-repair boundary 内，clean/noisy pairs make the correction direction observable. The contribution is not a new optimization trick, but identifying and exploiting a system-level interface where a lightweight repair can recover frozen-tracker performance.

这句话很重要。它把简单性转化为优点：

因为问题被正确切开，所以方法可以简单。

这就是你说的 mixup 型顶会工作背后的逻辑：不是技术复杂，而是切入角度让一个简单操作有了高杠杆。

所以现在的目标不是让 HSL 看起来高级，而是让审稿人觉得：

“啊，motion tracking robustness 里确实有一类 failure 被归错因了。原来不是一定要 train a stronger recovery policy，有时应该先 repair the reference interface。”

如果能让审稿人产生这个感觉，文章就有机会。

### User

你的解释让我有了更深的理解：我们的架构这样设计，其实本身就是因为我们已经意识到了问题需要被这样解决，因此我们只要清晰直白得解释为什么这样设计就行了，而且这个思路让HSL很清晰得与更强力的HRL区分开了，既然我们要做Repaire，那很明显就不应该搞rho，因为rho代表了Recovery的接受度，那么HSL就是个很自然的选择，而且我们这个工作只是证明这个问题的确存在，HSL服务于这个目的，如果使用HRL和rho，那问题就混杂在一起了，没法清晰切分开

### Assistant

对，你这段理解已经很接近“论文主心骨”了，而且比我们之前所有 rho/critic 讨论都更干净。

最关键的一句是：

既然本文研究的是 repair，就不应该把 recovery 的接受度混进来。

这一下把 HSL 从“低级方法”变成了“边界正确的方法”。

因为在 repair 问题里，我们的假设是：

Clean reference 本身仍然可执行，只是当前输入 reference 被污染。

在这个假设下，目标不是判断“当前状态能不能接受这段 motion”，而是判断“这个 corrupted reference 应该如何被拉回 clean reference”。这个方向是可观测的，所以 HSL 是自然的。它不是退而求其次，而是问题定义后的直接解。

rho 的角色不同。rho 问的是：

这个 repair 在当前动态状态下能不能被执行，或者应该执行多少？

这已经在问 recovery/acceptance。它需要 state-conditioned dynamic evidence，需要 rollout comparison，需要更复杂的 credit assignment。它不是 repair 的必要变量。

所以如果我们把 rho 放进当前文章，会发生两个问题：

1. 概念混杂  
   文章一边说自己修 reference，一边又学习 dynamic acceptance。审稿人会不知道你到底是在做 reference repair，还是 recovery control。

2. 证据链变复杂  
   HSL 只需要 clean/noisy reference pair 就能构造监督信号。rho 需要 rollout、baseline、critic、k-step、warmup。复杂度上去了，但反而削弱了“问题被清楚切开”的美感。

这就是你刚才说的：HSL 服务于证明问题存在。  
更准确地说，HSL 服务于证明：

reference-side corruption 是一个可以被独立修复的 failure source。

如果这个证明成立，文章贡献就不需要 rho。rho 是下一层问题：当 repair 方向存在，但当前身体状态未必能接受时，如何做 dynamic acceptance。

我会把当前文章边界写成这样：

This work studies reference repair, not recovery. We assume the clean reference remains executable by the frozen tracker, and learn a task-space residual that maps corrupted references back toward this executable reference interface. Under this boundary, the correction direction is observable from clean/noisy pairs, making supervised Delta SE(3) learning the appropriate minimal mechanism.

这句话非常重要。它解释了三件事：

- 为什么不是 recovery；
- 为什么不是 rho；
- 为什么 HSL 是合理的。

你也可以在内部给自己一个更简单的版本：

Repair 看 reference，Recovery 看 state。

Repair 的训练信号来自 clean/noisy reference pair。  
Recovery 的训练信号来自 state transition / rollout outcome。  
FrontRES-HSL 属于前者。rho/critic 属于后者。

这样边界就稳了。

而且这个切法还能帮你处理审稿人问题。  
如果审稿人问“为什么不用 RL/HRL？”  
你可以回答：

因为在 reference repair setting 中，the desired correction is directly observable from clean/noisy references. RL would entangle reference repair with downstream recovery behavior and introduce unnecessary credit assignment noise. We use supervised task-space residual learning to isolate and test the reference-interface hypothesis.

这不是怕 RL，而是故意不用 RL，以保证问题干净。

如果审稿人问“为什么不处理 high disturbance？”  
你可以回答：

High disturbance often enters dynamic recovery, where even the clean reference may not be immediately admissible from the current robot state. This requires learning state-conditioned acceptance or recovery behavior, which is complementary to our reference repair layer.

这也不是逃避，而是边界清楚。

我觉得你现在已经从“如何让 HSL 显得不 low”走到了更好的位置：

HSL 不需要显得高级。它需要显得必然。

当审稿人觉得“在这个问题定义下，确实就该这么做”，方法就成立了。

这也是好切入角度的力量：它让一个简单方法看起来不是简单，而是准确。

### User

我认为讨论得非常清楚了，现在请将我们的对话输出到./note的文档中，请新建个文档，尽量详细得整理讨论内容，我将依据这个文档为工作进行收尾

### Assistant

Created `note/FrontRES Repair Paper Closure.md` as a structured discussion record.


