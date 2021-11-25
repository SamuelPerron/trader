from enum import Enum


def log(message):
    with open('logs.txt', 'a') as file:
        file.write(str(message) + '\n')


class CoolEnum(Enum):
    @classmethod
    def values(cls) -> list:
        return [choice.value for choice in list(cls)]

    @classmethod
    def must_be_complex(cls) -> None:
        if type(cls.values()[0]) != tuple:
            raise NotImplementedError()

    @classmethod
    def values_short(cls) -> list:
        cls.must_be_complex()

        return [choice[0] for choice in cls.values()]

    @classmethod
    def values_long(cls) -> list:
        cls.must_be_complex()

        return [choice[1] for choice in cls.values()]

    @classmethod
    def short_to_long(cls, short: str) -> str:
        cls.must_be_complex()

        for choice in cls.values():
            if choice[0] == short:
                return choice[1]