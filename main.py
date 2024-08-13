from app.main.consumer.pipeline_consumer import PipelineConsumer
from app.main.util.register import init_processors

if __name__ == '__main__':
    processors = init_processors()
    pipeline_consumer = PipelineConsumer(processors)
    pipeline_consumer.main()
