#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Template pools for brain MRI multi-round perception conversations.

The builder keeps anatomy/mask selection logic in 3_build_multiround_dataset.py.
This file owns the dialog categories and natural-language variants so prompt
diversity can be improved without touching dataset construction code.
"""

from __future__ import annotations

from collections import Counter
from typing import Any, Mapping, MutableMapping


DIALOGUE_CATEGORY_ORDER = [
    "basic_segmentation",
    "tissue_to_region",
    "contralateral_same_region",
    "same_side_same_lobe",
    "spatial_named_region",
    "tumor_to_overlapping_region",
]

DIALOGUE_CATEGORY_DESCRIPTIONS = {
    "basic_segmentation": "One-round direct segmentation dialogs for improving image-to-mask grounding without relational reasoning.",
    "tissue_to_region": "Use a tissue mask as the first-round visual cue, then ask for a fine anatomical region with matching tissue attributes.",
    "contralateral_same_region": "Use one named anatomical region as a reference, then ask for its contralateral counterpart.",
    "same_side_same_lobe": "Use one same-side local anatomical structure as context, then ask for another visible region in the same lobe or structural scope.",
    "spatial_named_region": "Use one visible region as a 2D spatial anchor, then ask for a named region in a relative image direction.",
    "tumor_to_overlapping_region": "Use a BraTS tumor-related abnormality mask as a lesion cue, then ask for the overlapping BrainParc anatomical region.",
}

DIRECTION_ZH = {
    "left": "左方",
    "right": "右方",
    "up": "上方",
    "down": "下方",
}

TEMPLATE_POOLS = {
    "region_round1": [
        "请分割当前切片中的{region}。",
        "请在这张切片上标出{region}。",
        "请定位并分割当前图像里的{region}。",
        "请找出当前二维切片中的{region}并完成分割。",
        "请识别该切片上的{region}。",
        "请在当前MRI切片中勾画{region}。",
        "请将图像中可见的{region}分割出来。",
        "请对当前切片内的{region}进行区域标注。",
        "请在这幅图像中提取{region}对应的解剖区域。",
        "请判断{region}在当前切片中的位置，并进行分割。",
        "请只分割当前图像中属于{region}的区域。",
        "请完成当前切片上{region}的精确分割。",
        "请观察该MRI层面，并给出{region}的分割结果。",
        "请在当前视野内寻找{region}，然后输出其掩膜。",
        "请把这张图像中可辨认的{region}单独标注出来。",
        "请依据解剖边界分割当前切片上的{region}。",
        "请对{region}做一次像素级定位与分割。",
        "请在不包含相邻结构的前提下分割{region}。",
    ],
    "tissue_round1": [
        "请分割当前切片中的{tissue}。",
        "请标出这张切片上的{tissue}。",
        "请在当前二维图像中分割{tissue}。",
        "请识别并分割当前切片中的{tissue}。",
        "请在该MRI切片中勾画{tissue}。",
        "请提取图像中可见的{tissue}部分。",
        "请对当前图像内的{tissue}进行像素级标注。",
        "请将当前切片中属于{tissue}的区域分割出来。",
        "请定位这张图像上的{tissue}并完成分割。",
        "请只标注当前切片中符合{tissue}属性的区域。",
        "请根据组织信号特征分割当前图像中的{tissue}。",
        "请完成当前二维MRI图像中{tissue}的区域分割。",
        "请在当前层面提取所有可见的{tissue}。",
        "请把图像中呈现为{tissue}类别的区域标注出来。",
        "请从组织类别角度分割当前切片内的{tissue}。",
        "请给出该切片上{tissue}的完整掩膜。",
    ],
    "basic_region_segmentation": [
        "请分割当前切片中的{region}。",
        "请在当前MRI图像中标注{region}。",
        "请对{region}进行像素级分割。",
        "请定位并分割这张切片上的{region}。",
        "请只输出{region}对应的区域掩膜。",
        "请把当前图像中可见的{region}勾画出来。",
        "请识别{region}的边界并完成分割。",
        "请在不包含邻近脑区的情况下分割{region}。",
        "请提取当前层面内的{region}。",
        "请给出{region}在该切片中的完整分割结果。",
        "请根据解剖结构名称分割{region}。",
        "请在这张二维MRI切片上找出并标注{region}。",
        "请生成{region}的二值分割掩膜。",
        "请将图像中属于{region}的像素标记出来。",
        "请完成一次直接的{region}分割。",
        "请针对当前切片分割目标脑区：{region}。",
        "请勾画{region}，并避免包含背景或其他结构。",
        "请根据当前图像内容分割指定解剖区域{region}。",
        "请从整张切片中提取{region}对应区域。",
        "请标出所有可见的{region}像素。",
    ],
    "basic_tissue_segmentation": [
        "请分割当前切片中的{tissue}。",
        "请对{tissue}进行像素级标注。",
        "请在当前MRI图像中提取{tissue}。",
        "请只输出{tissue}对应的分割掩膜。",
        "请把这张切片上的{tissue}完整分割出来。",
        "请根据组织类别标注{tissue}。",
        "请定位当前层面的{tissue}并完成分割。",
        "请生成{tissue}的二值掩膜。",
        "请将图像中属于{tissue}的区域标记出来。",
        "请完成当前二维切片上的{tissue}分割。",
        "请在不包含其他组织类别的前提下分割{tissue}。",
        "请识别并勾画所有可见的{tissue}。",
        "请直接分割目标组织：{tissue}。",
        "请提取当前图像中的{tissue}像素区域。",
        "请给出该切片上{tissue}的完整区域标注。",
    ],
    "basic_tumor_segmentation": [
        "请分割当前切片中的{tumor}。",
        "请在当前MRI图像中标注{tumor}。",
        "请对{tumor}进行像素级分割。",
        "请定位并分割这张切片上的{tumor}。",
        "请只输出{tumor}对应的区域掩膜。",
        "请把当前图像中可见的{tumor}勾画出来。",
        "请提取该层面内的{tumor}。",
        "请给出{tumor}在该切片中的完整分割结果。",
        "请生成{tumor}的二值分割掩膜。",
        "请将图像中属于{tumor}的像素标记出来。",
        "请完成一次直接的{tumor}分割。",
        "请针对当前切片分割目标异常区域：{tumor}。",
    ],
    "tissue_to_region": [
        "请基于实例1所代表的组织类别，分割当前切片中与该组织属性对应的{region}。",
        "实例1给出了组织属性线索，请进一步分割属于该组织属性的{region}。",
        "参考实例1的组织类别，请找出并分割对应的{region}。",
        "请根据实例1所提示的组织类型，在当前切片中分割{region}。",
        "实例1展示了相关组织属性，请据此定位并分割{region}。",
        "请利用实例1的组织类别信息，找出当前图像中的{region}。",
        "请以实例1的组织属性为参照，分割具有相同组织特征的{region}。",
        "实例1提供了组织层面的参考，请在当前切片中标注{region}。",
        "请结合实例1所示组织类型，完成{region}的区域分割。",
        "请根据实例1对应的组织信号线索，识别并分割{region}。",
        "请从与实例1组织属性一致的结构中，分割出{region}。",
        "请参考实例1的组织类别提示，在当前二维图像中提取{region}。",
        "实例1可作为组织属性示例，请据此分割当前切片中的{region}。",
        "请根据实例1所代表的组织成分，定位当前图像内的{region}。",
        "请沿用实例1提供的组织类别先验，分割目标区域{region}。",
        "实例1标出了一个组织类别，请在同类组织背景下寻找并分割{region}。",
        "请把实例1作为组织属性参照，进一步标出{region}。",
        "请依据实例1体现的组织归属，完成{region}的分割。",
        "实例1说明了目标可能所属的组织成分，请分割其中的{region}。",
        "请从实例1提示的组织范围里识别解剖目标{region}。",
    ],
    "contralateral_same_region": [
        "请分割与实例1位于对侧、且解剖名称对应的{region}。",
        "请根据实例1的同名对侧关系，分割{region}。",
        "实例1是参照结构，请分割其对侧对应的{region}。",
        "请以实例1为解剖参照，找到并分割对侧的{region}。",
        "请在实例1的镜像对侧位置，分割对应的{region}。",
        "请根据左右对称关系，分割与实例1相对应的{region}。",
        "实例1提示了一侧结构，请标出另一侧对应的{region}。",
        "请寻找实例1的对侧同源解剖结构，并分割{region}。",
        "请参考实例1的位置和名称，分割当前切片中的对侧{region}。",
        "请基于实例1的半球侧别，分割另一侧的{region}。",
        "请不要重复分割实例1，而是分割其对侧对应的{region}。",
        "请根据实例1所示结构，在对侧半球中定位并分割{region}。",
        "请利用实例1提供的同名结构线索，完成对侧{region}分割。",
        "请在与实例1相反侧的解剖区域中分割{region}。",
        "请识别实例1的对侧配对结构，并将{region}分割出来。",
        "实例1给出了左右配对中的一侧，请补充分割另一侧的{region}。",
        "请沿脑半球对称关系，从实例1推断并标注{region}。",
        "请把实例1作为同名结构参考，定位相反侧的{region}。",
        "请寻找与实例1名称匹配但侧别相反的{region}并分割。",
        "请根据实例1的解剖形态和侧别，分割对侧配对区域{region}。",
    ],
    "same_side_same_lobe": [
        "请参考实例1所在的{scope_phrase}，分割另一个可见的{region}。",
        "在实例1对应的{scope_phrase}内，请继续分割{region}。",
        "请沿用实例1的同侧解剖范围线索，分割{scope_phrase}中的{region}。",
        "请根据实例1所在的解剖范围，在同一{scope_phrase}内分割{region}。",
        "实例1提示了目标所在的局部范围，请在该{scope_phrase}中找出{region}。",
        "请不要跨越实例1所在的{scope_phrase}，仅在该范围内分割{region}。",
        "请以实例1的同侧局部区域为参照，分割可见的{region}。",
        "请在实例1所属的{scope_phrase}中定位并标注{region}。",
        "请依据实例1提供的同侧范围线索，完成{region}分割。",
        "实例1位于目标相关的{scope_phrase}内，请在相同范围中分割{region}。",
        "请保持与实例1相同的解剖侧别和范围，分割{region}。",
        "请利用实例1所在{scope_phrase}的信息，寻找并分割另一个{region}。",
        "请在实例1附近且属于同一{scope_phrase}的区域内分割{region}。",
        "请根据实例1的局部解剖上下文，分割同一{scope_phrase}内的{region}。",
        "请参考实例1限定的解剖范围，标出当前切片中的{region}。",
        "实例1给出同侧解剖背景，请在该背景内继续分割{region}。",
        "请把实例1作为局部范围锚点，选择并分割同范围的{region}。",
        "请结合实例1的侧别和结构分组，标注{region}。",
        "请在实例1同侧、同一结构范围中寻找{region}并输出掩膜。",
        "请利用实例1建立局部解剖上下文，然后分割{region}。",
    ],
    "spatial_named_region": [
        "请参考实例1在当前二维图像中的位置，分割位于其图像{direction_zh}的{region}。",
        "以实例1为位置参照，请分割图像{direction_zh}可见的{region}。",
        "请根据实例1的图像空间位置，找到并分割其{direction_zh}的{region}。",
        "请以实例1为中心参照，分割位于其{direction_zh}方向的{region}。",
        "请观察实例1的位置，并在图像{direction_zh}寻找{region}进行分割。",
        "请根据与实例1的相对空间关系，分割{direction_zh}侧的{region}。",
        "实例1提供了位置参考，请标出其图像{direction_zh}的{region}。",
        "请不要分割实例1本身，而是分割其{direction_zh}方向上的{region}。",
        "请在实例1相邻的图像{direction_zh}区域中定位并分割{region}。",
        "请结合实例1的二维空间位置，提取位于{direction_zh}的{region}。",
        "请根据实例1的方位线索，完成图像{direction_zh}侧{region}的分割。",
        "请从实例1出发，沿图像{direction_zh}方向寻找并分割{region}。",
        "请以实例1作为空间锚点，标注其{direction_zh}方向可见的{region}。",
        "请判断{region}相对于实例1的{direction_zh}位置，并完成分割。",
        "请基于实例1的相对方位，在当前切片中分割{region}。",
        "实例1是空间参照点，请定位图像{direction_zh}的{region}。",
        "请利用实例1的二维坐标关系，分割其{direction_zh}侧的{region}。",
        "请在实例1所处位置的{direction_zh}方向查找{region}并标注。",
        "请根据实例1与目标之间的图像方位关系，分割{region}。",
        "请围绕实例1建立空间参照，提取{direction_zh}方向上的{region}。",
    ],
    "tumor_round1": [
        "请分割当前切片中的{tumor}。",
        "请标出这张图像上的{tumor}。",
        "请识别并分割当前MRI切片内的{tumor}。",
        "请提取该层面可见的{tumor}。",
        "请对当前切片中的{tumor}做像素级标注。",
        "请在图像中勾画{tumor}的完整范围。",
        "请定位当前层面的{tumor}并输出掩膜。",
        "请将当前图像中属于{tumor}的部分分割出来。",
    ],
    "tumor_to_overlapping_region": [
        "请基于实例1的位置，分割与其空间重叠面积最大的{region}。",
        "实例1给出了病灶相关区域，请标出与其重叠最明显的{region}。",
        "请参考实例1的异常区域范围，分割重叠最多的{region}。",
        "请根据实例1的空间覆盖，定位并分割对应重叠最大的{region}。",
        "实例1是病灶空间线索，请进一步分割与其交叠最多的{region}。",
        "请找出实例1主要落入的解剖区域，并分割{region}。",
        "请依据实例1和BrainParc区域的重叠关系，分割{region}。",
        "请在实例1覆盖范围附近，标注重叠面积最大的解剖目标{region}。",
        "请利用实例1的病灶位置，确定并分割最相关的{region}。",
        "请不要重复实例1本身，而是分割与实例1重叠最多的{region}。",
    ],
}

TEMPLATE_POOL_SIZES = {name: len(items) for name, items in TEMPLATE_POOLS.items()}


def stable_index(size: int, *keys: Any) -> int:
    if size <= 0:
        raise ValueError("Template pool is empty.")
    key = "|".join(str(k) for k in keys)
    return sum(ord(ch) for ch in key) % size


def choose_template(
    pool_name: str,
    *keys: Any,
    usage_counter: MutableMapping[tuple[str, int], int] | None = None,
) -> str:
    options = TEMPLATE_POOLS[pool_name]

    if usage_counter is None:
        return options[stable_index(len(options), pool_name, *keys)]

    usages = [usage_counter[(pool_name, i)] for i in range(len(options))]
    min_usage = min(usages)
    least_used_indices = [i for i, value in enumerate(usages) if value == min_usage]
    chosen_offset = stable_index(len(least_used_indices), pool_name, *keys)
    idx = least_used_indices[chosen_offset]
    usage_counter[(pool_name, idx)] += 1
    return options[idx]


def render_template(
    pool_name: str,
    context: Mapping[str, Any],
    *keys: Any,
    usage_counter: MutableMapping[tuple[str, int], int] | None = None,
) -> str:
    template = choose_template(pool_name, *keys, usage_counter=usage_counter)
    return template.format(**context)


def summarize_template_usage(
    usage_counter: Mapping[tuple[str, int], int] | Counter,
) -> dict[str, dict[str, int]]:
    summary: dict[str, dict[str, int]] = {}
    for pool_name, pool_size in TEMPLATE_POOL_SIZES.items():
        values = [int(usage_counter.get((pool_name, i), 0)) for i in range(pool_size)]
        summary[pool_name] = {
            "pool_size": int(pool_size),
            "used_templates": int(sum(1 for value in values if value > 0)),
            "min_usage": int(min(values) if values else 0),
            "max_usage": int(max(values) if values else 0),
        }
    return summary
