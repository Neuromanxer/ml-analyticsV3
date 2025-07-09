class VisualizationOut(BaseModel):
    caption: str
    image_src: str  # URL that your frontend can fetch
class VisualizationPayload(BaseModel):
    visualizations: Dict[str, str]  # caption → base64 string

class VisualizationResponse(BaseModel):
    id: int
    caption: str
    image_src: str
    created_at: datetime

    class Config:
        # V2 replacement for orm_mode
        from_attributes = True
