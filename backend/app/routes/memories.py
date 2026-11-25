"""The calling user's remembered facts: list and delete. 404 when memory is off."""

from typing import Annotated

from chatbot.store.memory_store import Memory, UserMemoryStore
from fastapi import APIRouter, Depends, HTTPException, Response

from app.auth import CurrentUser
from app.deps import RuntimeDep

router = APIRouter(prefix="/memories", tags=["memories"])
MAX_LISTED = 500


def memory_store(runtime: RuntimeDep) -> UserMemoryStore:
    memories = runtime.chatbot.memories
    if memories is None:
        raise HTTPException(status_code=404, detail="Memory is disabled")
    return memories


MemoryStoreDep = Annotated[UserMemoryStore, Depends(memory_store)]


@router.get("")
async def list_memories(user: CurrentUser, memories: MemoryStoreDep) -> list[Memory]:
    page, _ = await memories.list(user.id, limit=MAX_LISTED)
    return page


@router.delete("/{memory_id}", status_code=204)
async def delete_memory(memory_id: int, user: CurrentUser, memories: MemoryStoreDep) -> Response:
    if not await memories.delete(user.id, memory_id):
        raise HTTPException(status_code=404, detail="Memory not found")
    return Response(status_code=204)
