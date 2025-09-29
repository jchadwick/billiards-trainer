import React from 'react'
import { observer } from 'mobx-react-lite'
import { Button as BaseButton, ButtonProps as BaseButtonProps } from '../ui/Button'

export interface FormButtonProps extends BaseButtonProps {
  formType?: 'submit' | 'reset' | 'button'
  loading?: boolean
  loadingText?: string
  confirmationRequired?: boolean
  confirmationText?: string
  onConfirm?: () => void
}

export const FormButton = observer<FormButtonProps>(({
  formType = 'button',
  loading = false,
  loadingText,
  confirmationRequired = false,
  confirmationText = 'Are you sure?',
  onConfirm,
  children,
  onClick,
  disabled,
  ...props
}) => {
  const [showConfirmation, setShowConfirmation] = React.useState(false)

  const handleClick = (e: React.MouseEvent<HTMLButtonElement>) => {
    if (confirmationRequired && !showConfirmation) {
      e.preventDefault()
      setShowConfirmation(true)
      return
    }

    if (showConfirmation) {
      setShowConfirmation(false)
      onConfirm?.()
    }

    onClick?.(e)
  }

  const isDisabled = disabled || loading

  const buttonContent = loading && loadingText ? loadingText : children

  if (confirmationRequired && showConfirmation) {
    return (
      <div className="flex items-center gap-2">
        <span className="text-sm text-secondary-600 dark:text-secondary-400">
          {confirmationText}
        </span>
        <BaseButton
          size="sm"
          variant="danger"
          onClick={handleClick}
          {...props}
        >
          Confirm
        </BaseButton>
        <BaseButton
          size="sm"
          variant="ghost"
          onClick={() => setShowConfirmation(false)}
        >
          Cancel
        </BaseButton>
      </div>
    )
  }

  return (
    <BaseButton
      type={formType}
      loading={loading}
      disabled={isDisabled}
      onClick={handleClick}
      {...props}
    >
      {buttonContent}
    </BaseButton>
  )
})

FormButton.displayName = 'FormButton'

// Convenience components for common form button types
export const SubmitButton = observer<Omit<FormButtonProps, 'formType'>>(
  (props) => (
    <FormButton formType="submit" variant="primary" {...props} />
  )
)

SubmitButton.displayName = 'SubmitButton'

export const ResetButton = observer<Omit<FormButtonProps, 'formType'>>(
  (props) => (
    <FormButton formType="reset" variant="outline" {...props} />
  )
)

ResetButton.displayName = 'ResetButton'

export const CancelButton = observer<Omit<FormButtonProps, 'formType'>>(
  (props) => (
    <FormButton formType="button" variant="ghost" {...props} />
  )
)

CancelButton.displayName = 'CancelButton'

export const DeleteButton = observer<Omit<FormButtonProps, 'formType' | 'confirmationRequired'>>(
  (props) => (
    <FormButton
      formType="button"
      variant="danger"
      confirmationRequired={true}
      confirmationText="Are you sure you want to delete this item?"
      {...props}
    />
  )
)

DeleteButton.displayName = 'DeleteButton'
