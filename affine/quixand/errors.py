class QSAdapterError(Exception):
	pass


class QSSandboxNotFound(Exception):
	pass


class QSTimeout(Exception):
	pass


class QSProcessError(Exception):
	def __init__(self, message: str, exit_code: int | None = None):
		super().__init__(message)
		self.exit_code = exit_code


class QSFilesystemError(Exception):
	pass


class QSTemplateError(Exception):
	pass



class QSProxyError(Exception):
	pass


