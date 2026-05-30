"""将 data/ 目录下的示例文档导入知识库。"""

from pathlib import Path

from app.core.knowledge_base import KnowledgeBaseService
from app.core.logger import logger

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_DATA_DIR = PROJECT_ROOT / "data"


def import_data_directory(data_dir: Path | None = None) -> list[dict]:
    """批量导入 txt 文件，返回每条导入结果。"""
    data_dir = data_dir or DEFAULT_DATA_DIR
    if not data_dir.exists():
        raise FileNotFoundError(f"数据目录不存在: {data_dir}")

    txt_files = sorted(data_dir.glob("*.txt"))
    if not txt_files:
        raise FileNotFoundError(f"目录中没有 txt 文件: {data_dir}")

    service = KnowledgeBaseService()
    results = []
    for path in txt_files:
        logger.info(f"[Seed] 导入 {path.name} ...")
        text = path.read_text(encoding="utf-8")
        message = service.upload_by_str(text, path.name)
        results.append({"filename": path.name, "message": message})
        logger.info(f"[Seed] {path.name}: {message}")
    return results


if __name__ == "__main__":
    import sys

    target = Path(sys.argv[1]) if len(sys.argv) > 1 else DEFAULT_DATA_DIR
    items = import_data_directory(target)
    print(f"共处理 {len(items)} 个文件")
    for item in items:
        print(f"  - {item['filename']}: {item['message']}")
