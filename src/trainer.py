import os
import torch
import time
import logging
from torch.utils.data import Dataset, DataLoader
from torch.utils.tensorboard import SummaryWriter
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator
from tqdm import tqdm
from .metrics import calculate_ap
from .datasets import collator


def train(
    model: torch.nn.Module,
    device: torch.device,
    train_data: Dataset,
    val_data: Dataset,
    batch_size: int,
    eval_batch_size: int,
    total_steps: int,
    eval_steps: int,
    patience: int,
    learning_rate: float,
    optimizer: str,
    lr_scheduler: str,
    output_path: str,
):
    """训练模型

    Args:
        model: 模型
        device: 训练设备
        train_data: 训练集
        val_data: 验证集
        input_size: 输入大小
        output_size: 输出大小
        batch_size: 批大小
        eval_batch_size: 验证时的批大小
        total_steps: 总训练步数
        eval_steps: 评估步数，每 eval_steps 步评估一次模型
        patience: 连续 patience 次评估结果不提升时停止训练
        learning_rate: 学习率
        optimizer: 优化器
        lr_scheduler: 学习率调度器
        output_path: 输出路径
    """
    os.makedirs(output_path, exist_ok=True)
    logger = logging.getLogger("train")
    handler = logging.FileHandler(os.path.join(output_path, "train.log"), "w")
    handler.setLevel(logging.INFO)
    logger.addHandler(handler)
    logger.info(
        f"Batch size: {batch_size}, Total steps: {total_steps}, Evaluation steps: {eval_steps}, Patience: {patience}, "
        f"Learning rate: {learning_rate}, Optimizer: {optimizer}, LR scheduler: {lr_scheduler}, "
    )
    writer = SummaryWriter(os.path.join(output_path, "tensorboard"), filename_suffix="_train")

    # 定义数据加载器
    train_dataloader = DataLoader(train_data, batch_size=batch_size, collate_fn=collator, shuffle=True, num_workers=8, pin_memory=True)
    val_dataloader = DataLoader(val_data, batch_size=eval_batch_size, collate_fn=collator, shuffle=False, num_workers=8, pin_memory=True)

    # 选择优化器
    if optimizer == "SGD":
        optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate)
    elif optimizer == "Momentum":
        optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate, momentum=0.9)
    elif optimizer == "AdamW":
        optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)
    else:
        raise ValueError("optimizer must be one of SGD, Momentum or AdamW")

    # 选择学习率调度器
    if lr_scheduler == "LinearLR":
        lr_scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lambda step: 1 - step / total_steps)
    elif lr_scheduler == "OneCycleLR":
        lr_scheduler = torch.optim.lr_scheduler.OneCycleLR(optimizer, max_lr=learning_rate, total_steps=total_steps)
    elif lr_scheduler == "CosineAnnealingLR":
        lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=total_steps, eta_min=0, last_epoch=-1)
    elif lr_scheduler == "FixedLR":
        lr_scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lambda step: 1)
    else:
        raise ValueError("lr_scheduler must be one of Linear, OneCycleLR, CosineAnnealingLR or FixedLR")

    model.to(device)
    start_time = time.time()
    train_loss_list = []
    val_loss_list = []
    val_ap_list = []
    best_val_ap = 0
    patience_count = 0
    logger.info("Start Training")
    train_data_iter = iter(train_dataloader)
    for step in range(0, total_steps, eval_steps):
        logger.info(f"Step {step}/{total_steps}")
        # 训练
        model.train()
        train_loss = 0
        for _ in (pbar := tqdm(range(eval_steps), desc="Training")):
            optimizer.zero_grad()
            try:
                batch = next(train_data_iter)
            except StopIteration:
                train_data_iter = iter(train_dataloader)
                batch = next(train_data_iter)
            inputs = batch["inputs"].to(device)
            labels = batch["annotations"].to(device)
            loss = model(inputs, labels)["loss"]
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 0.1)
            optimizer.step()
            lr_scheduler.step()
            train_loss += loss.item()
            pbar.set_postfix_str(f"Loss: {loss.item():.4f}")
        train_loss /= eval_steps
        train_loss_list.append(train_loss)
        # 验证
        model.eval()
        val_loss = 0
        val_predictions = []
        val_annotations = []
        with torch.no_grad():
            for batch in tqdm(val_dataloader, desc="Validation"):
                inputs = batch["inputs"].to(device)
                labels = batch["annotations"].to(device)
                outputs = model(inputs, labels)
                # 将预测的边界框坐标缩放到原图上
                predictions = model.post_process(outputs, inputs.shape[-2:], threshold = 0.05)
                # 计算损失和保存结果
                val_loss += outputs["loss"].item() * len(inputs)
                val_predictions += predictions
                val_annotations += [label for label in labels]
        val_loss /= len(val_data)
        val_loss_list.append(val_loss)
        val_ap = calculate_ap(val_predictions, val_annotations, train_data.num_classes)
        val_ap = val_ap.mean()
        val_ap_list.append(val_ap)
        writer.add_scalars("loss", {"train_loss": train_loss, "val_loss": val_loss}, step)
        writer.add_scalar("val_ap", val_ap, step)
        writer.add_scalar("learning_rate", optimizer.param_groups[0]['lr'], step)
        logger.info(f"\tTraining loss: {train_loss:.4f}")
        logger.info(f"\tValidation loss: {val_loss:.4f}")
        logger.info(f"\tValidation AP: {val_ap:.4f}")
        logger.info(f"\tLearning rate: {optimizer.param_groups[0]['lr']}")
        # 保存最好的模型
        if val_ap > best_val_ap:
            best_val_ap = val_ap
            torch.save(model.state_dict(), os.path.join(output_path, "best_model.pt"))
            patience_count = 0
            logger.info("\tBest model saved!")
        else:
            patience_count += 1
            if patience_count == patience:
                logger.info("\tEarly stopping!")
                break
    logger.info(f"Training finished in {time.time() - start_time}s")

    # 绘制损失曲线
    plt.figure()
    plt.gca().xaxis.set_major_locator(MaxNLocator(integer=True)) # 设置横坐标为整数
    x_axis = [i * eval_steps for i in range(len(train_loss_list))]
    plt.plot(x_axis, train_loss_list, label="Training loss")
    plt.plot(x_axis, val_loss_list, label="Validation loss")
    plt.xlabel("Step")
    plt.ylabel("Loss")
    plt.legend()
    plt.savefig(os.path.join(output_path, "loss.png"))

    writer.close()